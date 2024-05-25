import math
import torch

# Flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0.
has_flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.heads == 0
        self.heads = config.heads
        self.dim = config.dim
        # Key, query, value projections for all heads, but in a batch.
        self.proj_in = torch.nn.Linear(config.dim, 3*config.dim, bias=config.bias)
        # Output projection
        self.proj_out = torch.nn.Linear(config.dim, config.dim, bias=config.bias)
        # Regularization
        self.dropout = config.dropout
        self.dropout_qk = torch.nn.Dropout(config.dropout)
        self.dropout_final = torch.nn.Dropout(config.dropout)
        self.causal_mask = config.causal_mask

    def forward(self, x):
        batch_size, max_seq_len, dim = x.size()

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim.
        q, k, v  = self.proj_in(x).split(self.dim, dim=2)
        k = k.view(batch_size, max_seq_len, self.heads, dim//self.heads).transpose(1,2) # (batch_size, heads, max_seq_len, head_dim)
        q = q.view(batch_size, max_seq_len, self.heads, dim//self.heads).transpose(1,2) # (batch_size, heads, max_seq_len, head_dim)
        v = v.view(batch_size, max_seq_len, self.heads, dim//self.heads).transpose(1,2) # (batch_size, heads, max_seq_len, head_dim)

        # Causal self-attention. Self-attend: (batch_size, heads, max_seq_len, head_dim) x (batch_size, heads, head_dim, max_seq_len) -> (batch_size, heads, max_seq_len, max_seq_len)
        if has_flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = attn.masked_fill(self.causal_mask[:,:,:max_seq_len,:max_seq_len] == 0, float("-inf"))
            attn = torch.nn.functional.softmax(attn, dim=-1)
            attn = self.dropout_qk(attn)
            y = attn @ v # (batch_size, heads, max_seq_len, max_seq_len) x (batch_size, heads, max_seq_len, head_dim) -> (batch_size, heads, max_seq_len, head_dim)
        y = y.transpose(1, 2).contiguous().view(batch_size, max_seq_len, dim) # Re-assemble all head outputs side by side.

        # Output projection
        y = self.dropout_final(self.proj_out(y))
        return y

    def _init_weights(self,resid_layers,mean,std):
        torch.nn.init.normal_(self.proj_in.weight,mean=mean,std=std)
        torch.nn.init.normal_(self.proj_out.weight,mean=mean,std=std/math.sqrt(resid_layers))

class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config.dim, config.hidden_factor*config.dim, bias=config.bias),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_factor*config.dim, config.dim, bias=config.bias),
            torch.nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

    def _init_weights(self,resid_layers,mean,std):
        torch.nn.init.normal_(self.layers[0].weight, mean=mean, std=std)
        # Apply special scaled init to the residual projections, as per GPT-2 paper.
        torch.nn.init.normal_(self.layers[2].weight, mean=mean, std=std/math.sqrt(resid_layers))

class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_in = torch.nn.LayerNorm(config.dim,bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_mlp = torch.nn.LayerNorm(config.dim,bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_in(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x

    def _init_weights(self,resid_layers,mean,std):
        self.attn._init_weights(resid_layers,mean,std)
        self.mlp._init_weights(resid_layers,mean,std)


class GPTConfig:
    def __init__(self,max_seq_len,vocab_size,dim,tblocks,heads,hidden_factor,dropout=0.0,bias=False):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.dim = dim
        self.tblocks = tblocks
        self.heads = heads
        self.hidden_factor = hidden_factor
        self.dropout = dropout
        self.bias = bias
        self.causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)

class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.te = torch.nn.Embedding(config.vocab_size, config.dim)
        self.pe = torch.nn.Embedding(config.max_seq_len, config.dim)
        self.dropout_emb = torch.nn.Dropout(config.dropout)
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(config) for _ in range(config.tblocks)])
        self.ln_f = torch.nn.LayerNorm(config.dim,bias=config.bias)
        self.linear_f = torch.nn.Linear(config.dim, config.vocab_size, bias=config.bias)
        # With weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions".
        # Not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.linear_f.weight = self.te.weight # https://paperswithcode.com/method/weight-tying

        torch.nn.init.normal_(self.te.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)
        for block in self.transformer_blocks:
            block._init_weights(resid_layers=2*self.config.tblocks,mean=0.0,std=0.02)
        # `self.linear_f.weight` is tied to `self.te.weight`. No need to init separately.

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pe.weight.numel()
        return n_params

    def forward(self, x, targets=None):
        batch_size, max_seq_len = x.size()
        assert max_seq_len <= self.config.max_seq_len, f"Cannot forward sequence of length {max_seq_len}, block size is only {self.config.max_seq_len}"
        pos = torch.arange(0, max_seq_len, dtype=torch.long, device=x.device) # shape (t)

        # Forward the GPT model itself.
        tok_emb = self.te(x) # Token embeddings of shape (b, t, dim).
        pos_emb = self.pe(pos) # Position embeddings of shape (t, dim).
        x = self.dropout_emb(tok_emb + pos_emb)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.linear_f(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), ignore_index=-1)
            #< logits.size(-1) = vocab_size
            #< targets.shape = (batch_size,max_seq_len)
            #< logits.shape = (batch_size,max_seq_len,vocab_size)
        else:
            # Inference-time mini-optimization: only forward the results on the very last position.
            logits = self.linear_f(x[:,[-1],:]) # note: using list [-1] to preserve the block dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # All weight tensors (including embeddings) decay. All biases don"t.
        decay_params = []
        nodecay_params = []
        self.decay_params_num = 0
        self.nodecay_params_num = 0
        for param in self.parameters():
            if param.requires_grad:
                if param.dim()<2:
                    nodecay_params.append(param)
                    self.nodecay_params_num += param.numel()
                else:
                    decay_params.append(param)
                    self.decay_params_num += param.numel()
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        self.decay_params_tcount = len(decay_params)
        self.nodecay_params_tcount = len(nodecay_params)

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)

        return optimizer

    # https://www.adamcasson.com/posts/transformer-flops
    @staticmethod
    def openai_flops_per_token(tblocks, heads, dim, max_seq_len, vocab_size, ff_ratio):
        """Open AI method for forward pass FLOPs counting of decoder-only Transformer."""
        d_attn = dim // heads
        d_ff = dim * ff_ratio

        embeddings = 4*dim
        attn_qkv = 2*tblocks*dim * 3*d_attn*heads
        attn_mask = 2*tblocks*max_seq_len * d_attn*heads
        attn_project = 2*tblocks * d_attn*heads * dim
        ff = 2*tblocks * 2*dim * d_ff
        logits = 2*dim*vocab_size

        return embeddings + attn_qkv + attn_mask + attn_project + ff + logits

    # Estimates the number of flops we do per iteration.
    def estimate_flops(self, fwdbwd_per_iter, dt):
        # See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        #flops_per_token = 6*self.get_num_params() + 12*(self.config.tblocks)*(self.config.heads)*(self.config.dim//self.config.heads)*(self.config.max_seq_len)
        flops_per_token = self.openai_flops_per_token(self.config.tblocks,self.config.heads,self.config.dim,self.config.max_seq_len,self.config.vocab_size,self.config.hidden_factor)
        flops_per_fwdbwd = flops_per_token * self.config.max_seq_len
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        return flops_per_iter * (1.0/dt) # per second

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices `x` (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you"ll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at `max_seq_len`.
            if x.size(1) > self.config.max_seq_len:
                x = x[:, -self.config.max_seq_len:]
            # Forward the model to get the logits for the index in the sequence.
            logits, losses = self(x)
            # Pluck the logits at the final step and scale by desired temperature.
            logits = logits[:,-1,:] / temperature
            # Optionally crop the logits to only the top k options.
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Apply softmax to convert logits to (normalized) probabilities.
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Sample from a multinomial distribution.
            x_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            x = torch.cat((x, x_next), dim=1)

        return x

    def crop_seq_len(self, max_seq_len):
        # Model surgery to decrease the block size if necessary.
        # E.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model.
        assert max_seq_len <= self.config.max_seq_len
        self.config.max_seq_len = max_seq_len
        self.pe.weight = torch.nn.Parameter(self.pe.weight[:max_seq_len])
        self.config.causal_mask = self.config.causal_mask[:,:,:max_seq_len,:max_seq_len]

    @staticmethod
    def from_pretrained(model_type, override_args=None):
        print("Error: \"GPT.from_pretrained\" is currently not implemented!")
        return 0
    #    assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    #    override_args = override_args or {} # default to empty dict
    #    # Only dropout can be overridden. See more notes below.
    #    assert all(k == "dropout" for k in override_args)
    #    from transformers import GPT2LMHeadModel
    #    print("loading weights from pretrained gpt: %s" % model_type)

    #    # `layers`, `n_head` and `dim` are determined from `model_type`.
    #    config_args = {
    #        "gpt2":         dict(layers=12, n_head=12, dim=768),  # 124M params
    #        "gpt2-medium":  dict(layers=24, n_head=16, dim=1024), # 350M params
    #        "gpt2-large":   dict(layers=36, n_head=20, dim=1280), # 774M params
    #        "gpt2-xl":      dict(layers=48, n_head=25, dim=1600), # 1558M params
    #    }[model_type]
    #    print("forcing vocab_size=50257, block_size=1024, bias=True")
    #    config_args["vocab_size"] = 50257 # Always 50257 for GPT model checkpoints.
    #    config_args["block_size"] = 1024 # Always 1024 for GPT model checkpoints.
    #    config_args["bias"] = True # Always True for GPT model checkpoints.
    #    # We can override the dropout rate, if desired.
    #    if "dropout" in override_args:
    #        print(f"overriding dropout rate to {override_args["dropout"]}")
    #        config_args["dropout"] = override_args["dropout"]
    #    # Create a from-scratch initialized minGPT model.
    #    config = GPTConfig(**config_args)
    #    model = GPT(config)
    #    sd = model.state_dict()
    #    sd_keys = sd.keys()
    #    sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # Discard this mask / buffer, not a param.

    #    # Init a huggingface/transformers model.
    #    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #    sd_hf = model_hf.state_dict()

    #    # Copy while ensuring all of the parameters are aligned and match in names and shapes.
    #    sd_keys_hf = sd_hf.keys()
    #    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] # ignore these, just a buffer
    #    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")] # same, just the mask (buffer)
    #    transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
    #    # Basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear.
    #    # This means that we have to transpose these weights when we import them.
    #    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    #    for k in sd_keys_hf:
    #        if any(k.endswith(w) for w in transposed):
    #            # Special treatment for the Conv1D weights we need to transpose.
    #            assert sd_hf[k].shape[::-1] == sd[k].shape
    #            with torch.no_grad():
    #                sd[k].copy_(sd_hf[k].t())
    #        else:
    #            # Vanilla copy over the other parameters.
    #            assert sd_hf[k].shape == sd[k].shape
    #            with torch.no_grad():
    #                sd[k].copy_(sd_hf[k])

    #    return model
