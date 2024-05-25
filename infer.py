# Funnily, without the following lines Python always gives "ModuleNotFoundError" if I try to import a local file, at least on my machine.
import os
import sys
sys.path.append(os.getcwd())
#< "ModuleNotFoundError" Fix End
import contextlib
import torch
import tiktoken
from model import GPT,GPTConfig
# -----------------------------------------------------------------------------
seed = 1337
device = "cuda" # "cpu", "cuda", "cuda:0", "cuda:1", etc.
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16" # "float32" or "bfloat16" or "float16".
compile = False # Use PyTorch 2.0 to compile the model to be faster.
init_from = "best" # "best", "latest", or a gpt2 variant (e.g. "gpt2-xl").
out_dir = "out" # Ignored if `init_from` is not "resume".
prompt = None
prompt_file = None
num_samples = 2
max_new_tokens = 512
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions.
top_k = 256 # Retain only the top_k most likely tokens, clamp others to have 0 probability.
exec(open("configurator.py", encoding="utf-8").read()) # Overrides from command line or config file.
# -----------------------------------------------------------------------------

#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu" # For later use in `torch.autocast`.
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
autocast = contextlib.nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Model
if (init_from=="best" or init_from=="latest"):
    ckpt_path = os.path.join(out_dir, f"{init_from}.pt")
    print(f"Init from \"{ckpt_path}\"...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    ## Fix the keys of the state dictionary :(
    ## Honestly no idea how checkpoints sometimes get this prefix. Have to debug more.
    #unwanted_prefix = "_orig_mod."
    #for k,v in list(state_dict.items()):
    #    if k.startswith(unwanted_prefix):
    #        print(f"unwanted prefix detected in the state dict: {k}")
    #        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    #model.load_state_dict(state_dict)
    ##< Fix-End
elif init_from.startswith("gpt2"):
    print(f"Init from \"{init_from}\"...")
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    print(f"Error: Unknown \"init_from\" value: {init_from}")
    print("--Now init from \"best\".")
model.to(device)
if compile:
    model = torch.compile(model) # Requires PyTorch 2.0 (optional).
model.eval()

# look for the meta pickle in case it is available in the dataset folder.
load_meta = False
if (init_from=="best" or init_from=="latest") and "config" in checkpoint and "dataset" in checkpoint["config"]: # Older checkpoints might not have these...
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    import pickle
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO Want to make this more general to arbitrary encoder/decoder schemes.
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    # OK, let"s assume gpt-2 encodings by default.
    print("No \"meta.pkl\" found, assuming GPT-2 encoding...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# Encode the beginning of the prompt.
if prompt_file is not None:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()
if prompt is not None:
    prompt_encoded = (torch.tensor(encode(prompt),device=device)[None,...]) # `[None,...]` adds a batch dimension.
else:
    prompt_encoded = torch.randint(low=0,high=50255,size=(1,1,),device=device)

# encode the beginning of the prompt
if prompt_file is not None:
    with open(prompt_file, "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]) # `[None, ...]` adds a batch dimension.

# run generation
with torch.no_grad():
    with autocast:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print("---------------Generation End---------------")

