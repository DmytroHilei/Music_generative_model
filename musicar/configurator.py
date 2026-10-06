"""
Poor man's configurator (from nanoGPT). Usage:
    python train.py config/pretrain.py --batch_size=8 --wandb_log=False
A bare argument is a python config file that is exec'd, --key=value overrides a global of train.py.
Exec'd from train.py, so it works on train.py's globals().
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        assert not arg.startswith('--'), f"expected --key=value, got {arg}"
        print(f"Overriding config with {arg}:")
        with open(arg) as f:
            print(f.read())
        exec(open(arg).read())
    else:
        assert arg.startswith('--'), f"expected --key=value, got {arg}"
        key, val = arg.split('=', 1)
        key = key[2:]
        if key not in globals():
            raise ValueError(f"Unknown config key: {key}")
        try:
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            attempt = val  # plain string
        if globals()[key] is not None and not isinstance(globals()[key], str):
            assert type(attempt) == type(globals()[key]) or (
                isinstance(globals()[key], float) and isinstance(attempt, int)
            ), f"type mismatch for {key}: {type(attempt)} vs {type(globals()[key])}"
        print(f"Overriding: {key} = {attempt}")
        globals()[key] = attempt
