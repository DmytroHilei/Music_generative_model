"""Run the engine: python -m musicar.serve [--checkpoint checkpoints/f1] [--port 8000]"""

import argparse
import sys

import uvicorn

from musicar.serve.app import create_app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--checkpoint', default='checkpoints/f1')
    ap.add_argument('--preset', action='append', default=None,
                    help='generate.py preset(s) every request starts from (default: f1)')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--name', default='laptop', help="the engine's name in /v1/health")
    ap.add_argument('--host', default='127.0.0.1', help='keep localhost until the token check exists')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # the generation log lines reach a redirected log as they happen
    app = create_app(args.checkpoint, args.preset or ['f1'], args.device, args.name)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
