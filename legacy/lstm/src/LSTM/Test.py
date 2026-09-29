import json
import numpy as np
import keras

# =====================
# Load model
# =====================
model = keras.models.load_model("LSTM_model.keras")

# =====================
# Load vocabulary
# =====================
with open("vocab.json", "r", encoding="utf-8") as f:
    token_to_id = json.load(f)

# We add PAD if not in tokens vocab
if "<PAD>" not in token_to_id:
    token_to_id["<PAD>"] = len(token_to_id)

id_to_token = {int(v): k for k, v in token_to_id.items()}

SEQ_LEN = 24
PAD_ID = token_to_id["<PAD>"]
UNK_ID = token_to_id["<UNK>"]

STRUCTURAL = {"<SONG>", "<INTRO>", "<VERSE>", "<PRECHORUS>", "<CHORUS>", "<BRIDGE>", "<OUTRO>", "<INTERLUDE>"}

# =====================
# Tokenization helper
# =====================
def tokenize_input(text):
    return text.strip().split()

# =====================
# Generation
# =====================
MAX_CHORDS_PER_BAR = 4  # max numner of chords between <BAR>

def generate(seed_text, steps=40, temperature=0.9):
    seed_tokens = tokenize_input(seed_text)
    seed_tokens = [t.upper() if t.startswith("<") else t for t in seed_tokens]
    seed_ids = [token_to_id.get(t, UNK_ID) for t in seed_tokens]

    if len(seed_ids) < SEQ_LEN:
        seed_ids = [PAD_ID] * (SEQ_LEN - len(seed_ids)) + seed_ids
    else:
        seed_ids = seed_ids[-SEQ_LEN:]

    generated_ids = seed_ids.copy()
    generated_tokens = [id_to_token[i] for i in seed_ids if i != PAD_ID]

    chords_in_bar = 0
    for t in reversed(generated_tokens):
        if t == "<BAR>":
            break
        if t not in STRUCTURAL:
            chords_in_bar += 1

    BAR_ID = token_to_id["<BAR>"]

    for _ in range(steps):
        x = np.array([generated_ids[-SEQ_LEN:]])
        probs = model.predict(x, verbose=0)[0]

        probs[PAD_ID] = 0.0
        probs[UNK_ID] = 0.0

        # If we exeed limit - add bar, because of maximum amount of chords between BARs
        if chords_in_bar >= MAX_CHORDS_PER_BAR:
            next_id = BAR_ID
            token = "<BAR>"
        else:
            # Dont allow last token to be some random chord, has to end with structural tokens
            last_token = generated_tokens[-1] if generated_tokens else None
            if last_token and last_token not in STRUCTURAL and last_token != "<BAR>":
                probs[token_to_id[last_token]] = 0.0

            logits = np.log(probs + 1e-9) / temperature
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            next_id = int(np.random.choice(len(probs), p=probs))

            token = id_to_token[next_id]
            if token in STRUCTURAL and chords_in_bar > 0:
                next_id = BAR_ID
                token = "<BAR>"

        if token in STRUCTURAL and generated_tokens and generated_tokens[-1] == token:
            continue

        if token == "<BAR>":
            chords_in_bar = 0
        elif token not in STRUCTURAL:
            chords_in_bar += 1

        generated_ids.append(next_id)
        generated_tokens.append(token)
    return generated_tokens

# =====================
# Example
# =====================
seed_text = "<SONG> <INTRO> Cm Gm <BAR>"

generated_tokens = generate(seed_text, steps=50, temperature=1)

output = " ".join(t for t in generated_tokens if t != "<PAD>")
print(output)