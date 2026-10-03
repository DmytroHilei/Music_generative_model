# Dress rehearsal of the cloud pipeline (cloud/README.md): config/long_450m.py with a tiny model and fast saves,
# so setup -> preflight -> launch -> sync -> stop -> host loss -> pull -> resume -> finish runs in minutes.
exec(open('config/long_450m.py').read())
n_layer = 4
n_embd = 256
n_head = 4
warmup_iters = 20
eval_interval = 100
eval_iters = 4
ckpt_interval_min = 0.5
