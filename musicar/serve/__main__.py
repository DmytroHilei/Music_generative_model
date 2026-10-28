"""Run the engine: python -m musicar.serve [--config config/serve/engine.toml] [--any-setting value] [--port 8000]"""

import argparse
import sys
from dataclasses import fields

import uvicorn

from musicar.serve.app import create_app
from musicar.serve.config import CONFIG, EngineConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--config', default=str(CONFIG), help='engine settings (TOML); the options below override it')
    for f in fields(EngineConfig):
        if f.name == 'presets':
            continue  # tables: edit the file
        if f.name == 'base':
            ap.add_argument('--base', action='append', help='generate.py preset(s) every request starts from')
        else:
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), help=f'(engine.toml: {f.name})')
    ap.add_argument('--host', default='127.0.0.1', help='keep localhost until the token check exists')
    ap.add_argument('--port', type=int, default=8000)
    args = vars(ap.parse_args())
    host, port, path = args.pop('host'), args.pop('port'), args.pop('config')
    sys.stdout.reconfigure(line_buffering=True)  # the generation log lines reach a redirected log as they happen
    uvicorn.run(create_app(EngineConfig.load(path, **args)), host=host, port=port)


if __name__ == '__main__':
    main()
