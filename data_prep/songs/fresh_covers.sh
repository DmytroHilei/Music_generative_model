#!/bin/bash
# New piano covers of validation-split songs -> covers val v5 (+VAL_SONGS songs) and a fresh covers test
# (data_prep/songs/prepare_fresh_covers.py). Steps, in order:
#   data_prep/songs/fresh_covers.sh fetch        # YouTube search + download, ~2 h (network only), log logs/fetch_fresh.log
#   data_prep/songs/fresh_covers.sh transcribe   # piano transcription on the GPU, ~27 s/song, log logs/transcribe_fresh.log
#   data_prep/songs/fresh_covers.sh split        # -> ukrainian_covers_v5.csv + ukrainian_covers_fresh_test.csv, prints the songs
#   data_prep/songs/fresh_covers.sh score CKPT   # fresh test: CKPT (the P3 finalist), overfit_adamw8_1e-4 @349, base, ua_2048_v2
# Env: VAL_SONGS (default 8), MAX_MINUTES (default 150), COOKIES='firefox:<profile dir>' if searches get 403s.
set -e
cd "$(dirname "$0")/../.."
OUT=data/covers_fresh
F=data/finetune/fresh
case "$1" in
fetch)
    [ -f $F/candidates.csv ] || .venv/bin/python data_prep/songs/prepare_fresh_covers.py candidates
    .venv-audio/bin/python data_prep/songs/fetch_songs.py --mode covers --only-known-songs --ukrainian-only \
        --songs-csv $F/candidates.csv --artists $F/artists.txt --output $OUT \
        --per-song 1 --per-artist 1000 --song-search-size 8 --search-sleep 15 --sleep 20 \
        --max-skips 3 --max-fails 5 --max-minutes "${MAX_MINUTES:-150}" \
        ${COOKIES:+--cookies-from-browser "$COOKIES"} 2>&1 | tee -a logs/fetch_fresh.log
    ;;
transcribe)
    .venv-audio/bin/python data_prep/songs/transcribe_covers.py --input $OUT --output $OUT 2>&1 \
        | tee -a logs/transcribe_fresh.log
    ;;
split)
    .venv/bin/python data_prep/songs/prepare_fresh_covers.py split --val-songs "${VAL_SONGS:-8}"
    ;;
score)
    [ -n "$2" ] || { echo "usage: $0 score <P3 finalist checkpoint>"; exit 1; }
    SC=".venv/bin/python eval/compare/seed_compare.py --csv data/finetune/ukrainian_covers_fresh_test.csv --seeds 1 2 3 4 --batch-size 14"
    $SC --checkpoints "$2" checkpoints/overfit_adamw8_1e-4/model_bf16_it349.pt checkpoints/long_450m_final \
        --settings T=0.9,dt=0.1,dens=prompt --out logs/fresh_test_450m.txt --json logs/fresh_test_450m.json
    $SC --checkpoints checkpoints/ua_2048_v2 --settings T=0.9 --out logs/fresh_test_v2.txt --json logs/fresh_test_v2.json
    tail -n +3 logs/fresh_test_450m.txt logs/fresh_test_v2.txt
    ;;
*)
    sed -n 2,9p "$0"; exit 1
    ;;
esac
