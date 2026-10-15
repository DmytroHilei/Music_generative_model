# data/

Datasets, token stores and their metadata. The scripts that build them live in `data_prep/`. Only the files marked
*tracked* are in git; everything else is downloaded or generated (paths below are relative to the repo root, where all
scripts run).

| Path | Contents | Made by |
|---|---|---|
| `combined.csv` (tracked) | MAESTRO + GiantMIDI-Piano file list (`split`, `midi_filename`); GiantMIDI split by composer | `data_prep/prepare_giantmidi.py --combine` |
| `styles.json` (tracked) | Style vocabulary (index = id, 0 = none). Tied to trained checkpoints: only append, never reorder | by hand |
| `artists/` (tracked) | Artist lists for the song / cover search (`songs.txt` round 1, `covers_*`, `round2_*`), `candidates.tsv` | by hand, `data_prep/artist_candidates.py` |
| `maestro-v3.0.0/` | MAESTRO v3 MIDI (year folders); `maestro-v3.0.0.csv/.json`, `LICENSE`, `README` tracked | `data_prep/download_maestro.py` |
| `giantmidi/` | GiantMIDI-Piano MIDI + `giantmidi.csv` | `data_prep/prepare_giantmidi.py` |
| `aria/` | Aria-MIDI metadata per store (`<name>.csv`: file_id, segment, split, genre, ...) | `data_prep/prepare_aria.py`, `subset_aria.py` |
| `gigamidi/` | GigaMIDI file list, note hashes and dedupe stats | `data_prep/prepare_gigamidi.py`, `dedupe_val.py` |
| `discover/` | Discover MIDI file list and progress | `data_prep/prepare_discover.py` |
| `midicaps/` | MidiCaps: `train.json` (168k captions + features, keyed by Lakh md5 = GigaMIDI file name), `midicaps.tar.gz` (Lakh MIDI, not unpacked). CC-BY-SA 4.0 | download from Hugging Face `amaai-lab/MidiCaps` |
| `xmidi/` | XMIDI: `XMIDI.zip` (108k `XMIDI_<emotion>_<genre>_<id>.midi`, not unpacked). No license stated: research use, don't redistribute | Google Drive link in `github.com/xmusic-project/XMIDI_Dataset` |
| `labels/` | Per store: `<store>.jsonl`, one line per file with a MidiCaps caption / XMIDI labels (`file` = index in the store, `md5`) | `data_prep/build_labels.py` |
| `audio/` | Downloaded studio songs, `<artist>/*.mp3` + `songs.csv` | `data_prep/fetch_songs.py`, `redownload.py` |
| `covers/`, `covers_fresh/` | Solo piano covers (mp3) and their transcriptions, `<artist>/midi/` | `data_prep/fetch_songs.py --mode covers`, `transcribe_covers.py` |
| `finetune/` | Piano reductions / multi-track MIDI per artist and the fine-tune CSVs (`ukrainian_*.csv`); `exclude.txt` tracked | `data_prep/audio_to_piano.py`, `prepare_finetune.py`, `prepare_fresh_covers.py` |
| `cache/` | Token stores `<name>_<split>[_<hash>]/` (`tokens.u16`, `programs.u8`, `offsets.npy`, `meta.json`) | `musicar/data_loader.py` (CSVs), the `data_prep/prepare_*` scripts (stores) |
| `hf_upload/` | Symlink staging folder + manifest for uploading stores to the Hugging Face Hub | `cloud/data.py` |
