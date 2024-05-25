import os
import datasets
import tiktoken
import numpy
import torch.utils
import torch.utils.data

load_from = "arrow"
#< "web" to download from huggingface.
#< "local" if you want to manually download the tar files, say if you want only a portion of the data to get some feelings about training a language model.
#< (Note: You have to put the tar files in the "dataset" dir, and set the `_N_DATA_FILES` variable in "dataset/build.py" accordingly if you only want a portion.)
#< "arrow" to load directly from the generated arrow files (in the huggingface cache).
#< Of course to obtain the arrow files you have to load from "web" of "local" at least once.
#< The "arrow" option is intented to save disk space. Technically the tar files have no use once the arrow files are generated.
generate_arrow_only = False

data_dir = "dataset" # Applicable only when `load_from`=="local" or "arrow". (Well, not exactly. The generated samples go to this dir.)
if os.getenv("DATASETS") is not None:
    data_dir = os.path.join(os.getenv("DATASETS"),"openwebtext")

arrow_files_total = 83 # Applicable only when `load_from=="arrow"`. For the full data it's 83. If you only have a portion of the data you have to check the number manually.
arrow_files_selected = arrow_files_total # Applicable only when `load_from=="arrow"`.
assert arrow_files_selected <= arrow_files_total

# Number of workers in `.map()` call.
# Good number to use is ~order number of `cpu cores // 2`.
num_proc = 16

# Number of workers in `load_dataset()` call
# Best number might be different from `num_proc` above as it also depends on NW speed.
# It is better than 1 usually though.
num_proc_load_dataset = num_proc

num_samples = 2 # Set this to a non-zero value if you want to have a look at the data. Output to "dataset/samples.txt".
num_samples_tokenized = 2 # Set this to a non-zero value if you want to have a look at the tokenized data. Output to "dataset/samples_tokenized.txt".
#< All samples are taken from the "val" split to reduce memory usage and fetching time.

shuffle_seed = None
test_size = 0.0005

enc = tiktoken.get_encoding("gpt2")

print_enc = True # Set to `True` if you want to examine the token value. Write to "./{enc.name}_enc.txt".

if __name__ == "__main__":
    if print_enc:
        with open(f"{enc.name}_enc.txt", mode="w", encoding="utf-8") as f:
            for i in range(enc.n_vocab):
                print(f"{i} - {enc.decode_single_token_bytes(i)}", file=f)

    # Takes 54GB in huggingface .cache dir, about 8M documents (8,013,769).

    if load_from == "web":
        dataset = datasets.load_dataset("openwebtext", trust_remote_code=True, num_proc=num_proc_load_dataset)
    elif load_from == "local":
        dataset = datasets.load_dataset(os.path.join(data_dir,"build.py"), trust_remote_code=True, num_proc=num_proc_load_dataset)
    elif load_from == "arrow":
        dataset = datasets.concatenate_datasets([datasets.Dataset.from_file(
            os.path.join(data_dir,f"build-train-{i:05d}-of-{arrow_files_total:05d}.arrow")) for i in range(arrow_files_selected)])
    else:
        print(f"Unknown `load_from` value \"{load_from}\". Defaulting to \"web\".")
        load_from = "web"
        dataset = datasets.load_dataset("openwebtext", trust_remote_code=True, num_proc=num_proc_load_dataset)

    # This results in:
    # >>> dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: ...
    #     })
    # })
    # Or if `load_from=="arrow"`:
    # Dataset({
    #     features: ['text'],
    #     num_rows: ...
    # })

    if not generate_arrow_only:
        # "owt" by default only contains the "train" split, so create a test split.
        if load_from != "arrow":
            split_dataset = dataset["train"].train_test_split(test_size=test_size, shuffle=True, seed=shuffle_seed)
        else:
            split_dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=shuffle_seed)
        split_dataset["val"] = split_dataset.pop("test") # Rename the test split to val.
        dataset = None

        # This results in:
        # >>> split_dataset
        # DatasetDict({
        #     train: Dataset({
        #         features: ['text'],
        #         num_rows: ...
        #     })
        #     val: Dataset({
        #         features: ['text'],
        #         num_rows: ...
        #     })
        # })

        with open(os.path.join(data_dir,"samples.txt"), mode="w", encoding="utf-8") as f:
            for _ in range(num_samples):
                number = numpy.random.randint(0,len(split_dataset["val"]))
                print(f"Fetching data number {number} (in \"val\" split) as sample... (Might take some time...)")
                print(f"-----Data Number {number} (in \"val\" split)-----",file=f)
                print(split_dataset["val"]["text"][number],file=f)
                print(f"-----Data Number {number} (in \"val\" split) End-----\n\n",file=f)

        # We now want to tokenize the dataset. First define the encoding function (gpt2 bpe).
        def process(example):
            ids = enc.encode_ordinary(example["text"]) # `encode_ordinary` ignores any special tokens.
            ids.append(enc.eot_token) # Add the end of text token, e.g. 50256 for gpt2 bpe.
            # Note: I think eot should be prepended not appended... Hmm. it"s called "eot" though...
            out = {"ids": ids, "len": len(ids)}
            return out

        # Tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="Tokenizing the splits....",
            num_proc=num_proc,
            # If I don't specify the file name, `map` will create new files every time I run this script, instead of reusing the files generated before.
            cache_file_names={
                "train":os.path.join(data_dir,f"train_tokenized_{enc.name}.arrow"),
                "val":os.path.join(data_dir,f"val_tokenized_{enc.name}.arrow")
            },
            # The default storage uses "int64" which really consumes too much space - the max token value is only 50256.
            features=datasets.Features({"ids": datasets.Sequence(feature=datasets.Value(dtype="uint16", id=None), length=-1, id=None), "len": datasets.Value(dtype="uint32", id=None)}),
        )

        # This results in:
        # >>> tokenized
        # DatasetDict({
        #     train: Dataset({
        #         features: ['ids', 'len'],
        #         num_rows: ...
        #     })
        #     val: Dataset({
        #         features: ['ids', 'len'],
        #         num_rows: ...
        #     })
        # })

        with open(os.path.join(data_dir,f"samples_tokenized_{enc.name}.txt"), mode="w", encoding="utf-8") as f:
            for _ in range(num_samples_tokenized):
                number = numpy.random.randint(0,len(tokenized["val"]))
                print(f"Fetching tokenized data number {number} (in \"val\" split) as sample... (Might take some time...)")
                print(f"-----Data (Tokenized) Number {number} (in \"val\" split)-----",file=f)
                data = tokenized["val"]["ids"][number]
                print(data,file=f)
                print(f"-----Data (Tokenized) Number {number} (in \"val\" split) End-----",file=f)
                print(f"Decoding data number {number} (in \"val\" split) as sample... (Might take some time...)")
                print(f"-----Data (Decoded) Number {number} (in \"val\" split)-----",file=f)
                print(enc.decode(data),file=f)
                print(f"-----Data (Decoded) Number {number} (in \"val\" split) End-----\n\n",file=f)

        split_dataset = None

        with open("test.txt", mode="w", encoding="utf-8") as f:
            for i in range(100):
                n = numpy.random.randint(0,len(tokenized["train"]))
                shard = tokenized["train"].shard(1024,n%1024)
                print(f"{n},{n%1024},{n//1024}")
                print(f"---------------{n},{n%1024},{n//1024}-------------------",file=f)
                print(shard["ids"][n//1024],file=f)
                print(f"==============={n},{n%1024},{n//1024}===================\n\n",file=f)

        #import tqdm

        ## Concatenate all the ids in each dataset into one large file we can use for training.
        #def write_binary(dsets,split,total_batches):
        #    dset = dsets[split]
        #    arr_len = numpy.sum(dset["len"], dtype=numpy.uint64)
        #    filename = os.path.join(data_dir, f"{split}.bin")
        #    dtype = numpy.uint16 # (Can do since `enc.max_token_value` == 50256 is < 2**16)
        #    if not os.path.isfile(filename):
        #        arr = numpy.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        #        idx = 0
        #        for batch_idx in tqdm.tqdm(range(total_batches), desc=f"writing {filename}"):
        #            # Batch together samples for faster write
        #            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
        #            arr_batch = numpy.concatenate(batch["ids"])
        #            # Write into mmap
        #            arr[idx : idx + len(arr_batch)] = arr_batch
        #            idx += len(arr_batch)
        #        arr.flush()

        #write_binary(tokenized,"train",1024) # Note: The batch size may have to be changed if you only have a portion of the data, i.e. the number of data is smaller than the batch size.
        #write_binary(tokenized,"val",1) # Note: Ditto.

        # To read the bin files later, e.g. with numpy:
        # m = numpy.memmap("train.bin", dtype=numpy.uint16, mode="r")
    #< if not generate_arrow_only:
#< if __name__ == "__main__":
