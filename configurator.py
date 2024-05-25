"""
Poor Man"s Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32.

The code in this file will be run as follows from e.g. train.py:
>>> exec(open("configurator.py", encoding="utf-8").read())

So it"s not a Python module, it"s just shuttling this code away from train.py.
The code in this script then overrides the `globals()`.

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if "=" not in arg:
        # Assume it's the name of a config file.
        assert not arg.startswith("--")
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file, encoding="utf-8") as f:
            print(f.read())
        exec(open(config_file, encoding="utf-8").read())
    else:
        # Assume it's a `--key=value` argument.
        assert arg.startswith("--")
        key, val = arg.split("=")
        key = key[2:]
        if key in globals():
            try:
                # Attempt to eval it it (e.g. if bool, number, or etc.).
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # If that goes wrong, just use `string`.
                attempt = val
            # Ensure the types match OK.
            assert type(attempt) == type(globals()[key])
            # Cross fingers.
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
