# Funnily, without the following lines Python always gives "ModuleNotFoundError" if I try to import a local file, at least on my machine.
import os
import sys
sys.path.append(os.getcwd())
#< "ModuleNotFoundError" Fix End
import contextlib
import numpy
import torch
import torch.distributed
# -----------------------------------------------------------------------------
seed = None # Set it if want strict reproducibility.
# DDP settings
backend = "nccl" # "nccl", "gloo", etc.
num_workers = 1
flops_promised = 44.1e12 # Set according to your own GPU, if you know it.
# system
device = "cuda" # "cpu", "cuda", "cuda:0", "cuda:1" etc., or try "mps" on macbooks.
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16" # "float32", "bfloat16", or "float16". "float16" will auto implement a GradScaler
compile = False # Use PyTorch 2.0 to compile the model to be faster.
#< (That said, on my machine the compiled version is only around 11% faster, so it's probably not worth it because I have to switch to Linux to enable compiling.)
# I/O
data_dir = "dataset"
out_dir = "out"
#
max_iters = 100000 # Total number of training iterations.
log_interval = 10
eval_interval = 300 # You might want to change the `patience` value (defined later) if you change this.
eval_only = False # If True, script exits right after the first eval.
always_save_checkpoint = True # If True, always saves a checkpoint after each eval.
init_from = "scratch" # "scratch" or "resume" or "gpt2*"
check_hyperparams = True # Whether to check if the hyperparameters are the same with `init_from=="resume"`.
early_stopping = True # If `False`, training will only end when `max_iters` is reached.
patience = 2 # The unit is `eval_interval` so in terms of iterations the patience value will be `patience*eval_interval`. Early stopping is triggered if `current_patience > patience``.
min_delta = 0.0 # If `best_val_loss-current_val_loss > min_delte`, then `best_val_loss` will be set to the current val loss. Note that this value has effect even without early stopping.
# Data
gradient_accumulation_steps = 6 # Used to simulate larger batch sizes.
mini_batch_size = 10 # Set according to your GPU memory.
batch_size = mini_batch_size*gradient_accumulation_steps
max_seq_len = 1024
eval_iters = 20*gradient_accumulation_steps
eval_mini_batch_size = mini_batch_size
# Model
dim = 768
tblocks = 12
heads = 12
hidden_factor = 4
dropout = 0.1 # For pretraining 0 is good, for finetuning try 0.1+.
bias = True # Do we use bias inside LayerNorm and Linear layers?
# AdamW optimizer
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip_norm = 1.0 # Clip gradients at this value, or disable with 0.
# Learning rate
import lr
lr_scheduler = lr.LinearDecay(min_lr=6e-5, max_lr=6e-4, warmup_iters=2000, decay_iters=100000)
#< `min_lr` Should be ~= `max_lr/10` per Chinchilla
# -----------------------------------------------------------------------------

# DDP init
ddp = int(os.environ.get("RANK", -1)) != -1 # is this a ddp run?
if ddp:
    torch.distributed.init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = (ddp_rank==0) # This process will do logging, checkpointing etc.
    seed_offset = ddp_rank # Each process gets a different seed.
    # `world_size` number of processes will be training simultaneously, so we can scale
     # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = ddp_world_size*batch_size*max_seq_len

def end_training(exitcode=0):
    if ddp:
        torch.distributed.destroy_process_group()
    sys.exit(exitcode)

import datetime
class Logger:
    def __init__(self):
        if master_process:
            if init_from=="resume":
                self.f = open(os.path.join(out_dir,"training.log"),mode="a", encoding="utf-8")
            else:
                self.f = open(os.path.join(out_dir,"training.log"),mode="w", encoding="utf-8")
        else:
            self.f = None
    def __call__(self,message="",end="\n",time=True):
        if time:
            now = datetime.datetime.now()
            #print(f"{now} -- ",file=sys.stdout,end="")
            if self.f is not None:
                print(f"{now} -- ",file=self.f,end="")
        print(message,file=sys.stdout,end=end,flush=True)
        if self.f is not None:
            print(message,file=self.f,end=end,flush=True)
    def __del__(self):
        if self.f is not None:
            self.f.close()

logger = Logger()

# Attempt to derive `vocab_size` from the dataset.
if init_from!="resume":
    import pickle
    meta_path = os.path.join(data_dir, "meta.pkl")
    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        logger(f"Found `vocab_size` = {meta_vocab_size} (inside {meta_path}).")
        vocab_size = meta_vocab_size
    except FileNotFoundError:
        logger(f"\"{meta_path}\" not found. Defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up to the nearest multiple of 64 for efficiency)")
        vocab_size = 50304

# Record all hyperparameters defined so far.
exec(open("configurator.py", encoding="utf-8").read()) # Overrides from command line or config file.
hyper_params = {k:v for k,v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str, lr.BaseLRScheduler))} # Will be useful for logging.
hyper_params.pop("init_from")
# -----------------------------------------------------------------------------

# Init these up here. Will load from chekcpoint if `init_from="resume"`.
iter_num = 0
best_val_loss = 1e9
history = []
current_patience = 0

if init_from=="scratch":
    logger("Training from scratch.")
    logger("-----Hyperparameters-----" ,time=False)
    for k,v in hyper_params.items():
        logger(f"{k} = {v}" ,time=False)
    logger("-----Hyperparameters End-----" ,time=False)
elif init_from == "resume":
    logger("\n",time=False)
    logger(f"Training from saved checkpoint.")
    if os.path.isfile(os.path.join(out_dir, "latest.pt")):
        checkpoint_path = os.path.join(out_dir, "latest.pt")
    elif os.path.isfile(os.path.join(out_dir, "best.pt")):
        checkpoint_path = os.path.join(out_dir, "best.pt")
    else:
        logger("Fatal Error: No checkpoint found!")
        end_training(1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    history = checkpoint["history"]
    logger(f"Finished loading checkpoint \"{checkpoint_path}\".")
    logger(f"With `iter_num`={iter_num}, `best_val_loss`={best_val_loss:.4f}, `current_patience`={current_patience}")
    iter_num += 1 # The checkpoint is saved before `iter_num` is incremented, so I have to increment it here.
    # Check hyperparameters.
    if check_hyperparams:
        hyper_params_saved = checkpoint["hyper_params"]
        same = True
        for k,v in hyper_params_saved.items():
            if v!=hyper_params[k]:
                same = False
                logger(f"Warning: The value of \"{k}\" is different from checkpoint!")
                logger(f"--Note: Value in checkpoint ({hyper_params_saved[k]}). Value this time ({hyper_params[k]}).")

        if not same:
            print("Different hyperparameters detected. Proceed anyway? (y/n)",file=sys.stderr)
            yn = input()
            if yn.startswith("y"):
                logger("Training with different hyperparameters this time.")
            else:
               logger("Job aborted due to different hyperparameter settings.")
               end_training(1)
    #< Check End
elif init_from.startswith("gpt2"):
    logger(f"Training from OpenAI GPT-2 weights: \"{init_from}\".")
else:
    logger(f"Error: Unknown \"init_from\" value: \"{init_from}\"!")
    logger("--Now training from scratch.")
    init_from = "scratch"

if seed is not None:
    torch.manual_seed(seed + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu" # For later use in torch.autocast.

# Note: float16 data type will automatically use a GradScaler.
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
# Initialize a GradScaler. If `enabled=False`, `scaler` is a no-op.
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=="float16"))
autocast = contextlib.nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Model init
from model import GPT,GPTConfig
model_conf = GPTConfig(max_seq_len=max_seq_len,vocab_size=vocab_size,dim=dim,tblocks=tblocks,heads=heads,hidden_factor=hidden_factor,dropout=dropout,bias=bias)
if init_from == "scratch":
    model = GPT(model_conf)
elif init_from == "resume":
    checkpoint_model_conf = checkpoint["model_conf"]
    # Force these attributes to be equal otherwise we can't even resume training.
    # The rest of the attributes (e.g. dropout) can stay as desired from command line.
    model_conf.max_seq_len = checkpoint_model_conf.max_seq_len
    model_conf.vocab_size = checkpoint_model_conf.vocab_size
    model_conf.dim = checkpoint_model_conf.dim
    model_conf.tblocks = checkpoint_model_conf.tblocks
    model_conf.heads = checkpoint_model_conf.heads
    model_conf.bias = checkpoint_model_conf.bias
    model = GPT(model_conf)
    state_dict = checkpoint["model"]
    ## Fix the keys of the state dictionary :(
    ## Honestly no idea how checkpoints sometimes get this prefix. Have to debug more.
    #unwanted_prefix = "_orig_mod."
    #for k,v in list(state_dict.items()):
    #    if k.startswith(unwanted_prefix):
    #        logger(f"Critical: Unwanted prefix detected in the state dict: {k}")
    #        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    ##< Fix-End
    model.load_state_dict(state_dict)
    # crop down the model block size if desired, using model surgery
    if max_seq_len < checkpoint_model_conf.max_seq_len:
        model.crop_seq_len(max_seq_len)
        model_conf.max_seq_len = max_seq_len  # so that the checkpoint will have the right value
elif init_from.startswith("gpt2"):
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # Read off the created config params, so we can store them into checkpoint correctly.
    model_conf.max_seq_len = model.config.max_seq_len
    model_conf.vocab_size = model.config.vocab_size
    model_conf.dim = model.config.dim
    model_conf.tblocks = model.config.tblocks
    model_conf.heads = model.config.heads
    model_conf.bias = model.config.bias

model.to(device)

# Optimizer
optimizer = model.configure_optimizers(weight_decay, lr_scheduler(0), (beta1, beta2))
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None # Free up memory?

if init_from!="resume":
    from model import has_flash
    if not has_flash:
        logger("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
    logger(f"Number of parameters: {model.get_num_params()/1e6:.2f}M.")
    logger(f"Num decayed parameter tensors: {model.decay_params_tcount}, with {model.decay_params_num:,} parameters.")
    logger(f"Num non-decayed parameter tensors: {model.nodecay_params_tcount}, with {model.nodecay_params_num:,} parameters.")

# Compile the model
if compile:
    logger("Compiling the model... (Takes some minutes)")
    model = torch.compile(model) # Requires PyTorch 2.0.
    logger("Compilation successful.")

# Wrap model into DDP container
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
    model_no_ddp = model.module # Unwrap DDP container if needed.
else:
    model_no_ddp = model

# Poor man"s data loader
printed = False
def get_batch(filename,batch_size,max_seq_len,eot=50256):
    # We recreate numpy.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data = numpy.memmap(os.path.join(data_dir,filename), dtype=numpy.uint16, mode="r")
    ix = torch.randint(low=0, high=len(data)-max_seq_len, size=(batch_size,))
    for i in range(batch_size):
        while ix[i]==eot:
            ix[i] = numpy.random.randint(0,len(data)-max_seq_len)
    x = torch.stack([torch.from_numpy((data[i:i+max_seq_len]).astype(numpy.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+max_seq_len+1]).astype(numpy.int64)) for i in ix])
    assert y.size(0)==batch_size
    assert y[0].size(0)==max_seq_len
    eot_encountered = False
    for i_block in range(batch_size):
        for i_token in range(max_seq_len):
            if eot_encountered:
                y[i_block][i_token] = -1
            elif y[i_block][i_token]==eot:
                last_eot = i_block
                eot_encountered = True
    #global printed
    #if (not printed) and eot_encountered:
    #    print(y[last_eot])
    #    printed = True
    if device_type == "cuda":
        # Pin arrays `x`,`y`, which allows us to move them to GPU asynchronously (`non_blocking=True`).
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Helps estimate an arbitrarily accurate loss over either split using many batches.
@torch.no_grad()
def estimate_loss(iters,mini_batch_size,val_only=True):
    out = {}
    model.eval()
    splits = ["val"]
    if val_only:
        out["train"] = float("nan")
    else:
        splits.append("train")
    for split in splits:
        losses = torch.zeros(iters)
        for k in range(iters):
            X, Y = get_batch(split+".bin",mini_batch_size,max_seq_len)
            with autocast:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if eval_only:
    losses = estimate_loss(eval_iters,eval_mini_batch_size)
    logger("Eval Only Mode:")
    logger(f"loss(train) {losses['train']:.4f}, loss(val) {losses['val']:.4f}")
    logger("Evaluation ended.")
    end_training(0)

# Training loop
import time
local_iter_num = 0 # Number of iterations in the lifetime of this process.
running_mfu = -1.0
X, Y = get_batch("train.bin",mini_batch_size,max_seq_len) # Fetch the very first batch.

logger("-----Training session begins.-----")
while iter_num < max_iters:
    t0 = time.time()

    # Determine and set the learning rate for this iteration.
    learning_rate = lr_scheduler(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    # Forward backward update, with optional gradient accumulation to simulate larger batch size.
    # And using the GradScaler if data type is `float16`.
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # In DDP training we only need to sync gradients at the last micro step.
            # The official way to do this is with model.no_sync() context manager, but
             # I really dislike that this bloats the code and forces us to repeat code.
            # Looking at the source of that context manager, it just toggles this variable.
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps-1)
        with autocast:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # Scale the loss to account for gradient accumulation.
        # Immediately async prefetch next batch while model is doing the forward pass on the GPU.
        X, Y = get_batch("train.bin",mini_batch_size,max_seq_len)
        # Backward pass, with gradient scaling if training in fp16.
        scaler.scale(loss).backward()

    # Clip the gradient, with unscaling if necessary.
    if grad_clip_norm!=0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

     # Step the optimizer and grad scaler if training in fp16.
    scaler.step(optimizer)
    scaler.update()
    # Flush the gradients as soon as we can. No need for this memory anymore.
    optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0

    if master_process:
        if iter_num % log_interval == 0:
            # Get loss as float. Note: this is a CPU-GPU sync point.
            # Scale up to undo the division above, approximating the true total loss (exact would have been a sum).
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num>=5 and flops_promised!=0: # Let the training loop settle a bit.
                mfu = model_no_ddp.estimate_flops(batch_size, dt) / flops_promised
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            logger(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, lr {learning_rate:.2e}, grad_norm {grad_norm:.2f}", end="")
            if(grad_norm > grad_clip_norm):
                logger(f"(clipped to {grad_clip_norm})", time=False)
            else:
                logger(time=False)
        # Evaluate the loss on train/val sets and write checkpoints.
        if iter_num>0 and iter_num%eval_interval==0:
            logger(f"-----Eval: ({eval_iters} * {eval_mini_batch_size})-----")
            losses = estimate_loss(eval_iters,eval_mini_batch_size)
            history.append((iter_num//eval_interval, iter_num, losses['train'], losses['val'], learning_rate))
            if (best_val_loss - losses["val"]) > min_delta:
                best_val_loss_next = losses["val"]
            else:
                best_val_loss_next = best_val_loss
            logger(f"iter {iter_num}: loss(train) {losses['train']:.4f}, loss(val) {losses['val']:.4f}, best_loss(val) {best_val_loss_next:.4f}")
            logger("-----Eval End-----")

            if compile:
                model_no_ddp_uncompiled = model_no_ddp._orig_mod
            else:
                model_no_ddp_uncompiled = model_no_ddp

            def save_checkpoint(filename):
                checkpoint = {
                    "model": model_no_ddp_uncompiled.state_dict(),
                    "model_conf": model_no_ddp.config,
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss_next,
                    "hyper_params": hyper_params,
                    "history": history,
                }
                logger("-----Saving checkpoint...-----")
                torch.save(checkpoint, os.path.join(out_dir, filename))
                logger(f"-----Checkpoint saved to \"{os.path.join(out_dir, filename)}\"-----")

            if (best_val_loss - losses["val"]) > min_delta:
                save_checkpoint("best.pt")
                latest_path = os.path.join(out_dir, "latest.pt")
                if os.path.isfile(latest_path):
                    os.remove(latest_path)
            elif always_save_checkpoint:
                save_checkpoint("latest.pt")

            if early_stopping:
                if (best_val_loss - losses["val"]) <= min_delta:
                    current_patience += 1
                    if current_patience > patience:
                        logger(f"-----Training ended with `patience` exceeded max({patience}).-----")
                        logger(f"-----`iter_num`={iter_num}, `best_val_loss`={best_val_loss:.4f}-----")
                        end_training(0)
                else:
                    current_patience = 0

            best_val_loss = best_val_loss_next
    iter_num += 1
    local_iter_num += 1

logger(f"-----Training ended with `iter_num` reached max({max_iters}).-----")
logger(f"-----`iter_num`={iter_num}, `best_val_loss`={best_val_loss:.4f}-----")
end_training(0)
