# Compute request

Everything written for the request for compute to train the 1B and 1.8B models.

| File | What it is |
|---|---|
| [`pitch.md`](pitch.md) | the one-page request |
| [`scaling/fit.md`](scaling/fit.md) | ladder of short runs and its fit (`runs.json`, `fit.json` next to it) |
| [`samples/`](samples/README.md) | style-picked clips: musicar-L continuations and musicar-L-caption text prompts; the same on [`/samples`](https://musicar-ivory.vercel.app/samples) |

Code and pages that produce the evidence stay in their usual places:

| Path | What it is |
|---|---|
| `config/pretrain/scale_ladder.py`, `tools/sweeps/scale_ladder.py` | the ladder runs |
| `tools/sweeps/scale_fit.py` | the fit, writes `scaling/` here |
| `eval/compare/h2h_listen.py` | listening clips and the tally of the blind test |
| `web/public/compare/` | the side-by-side page and the blind test (`/compare`, `/compare/index.html?test`) |
| `web/public/samples/` | the samples page (`/samples`), clips copied from `samples/` here |
