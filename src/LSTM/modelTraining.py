import tensorflow as tf
from tensorflow.keras.datasets import imdb
import numpy as np
import json

import math
import random

# =========================
# 0) Reproducibility
# =========================
SEED = 42 #???
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED) #???

# =========================
# 1) Load and split into songs
# =========================
TXT_PATH = "Dataset for LSTM.txt"

SECTION_TOKENS = {"<INTRO>", "<VERSE>", "<CHORUS>", "<BRIDGE>", "<INTERLUDE>", "<OUTRO>"}
SPECIAL_TOKENS = SECTION_TOKENS | {"<BAR>"}

with open(TXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

raw_songs = text.split("<SONG>")

songs = [s.strip() for s in raw_songs if s.strip()]

print("Songs found:", len(songs))

if len(songs) < 10:
    raise RuntimeError("Too few songs detected. Add blank lines between songs or keep title lines.")

# =========================
# 2) Train/Val/Test split by songs (IMPORTANT)
# =========================
random.shuffle(songs)

n = len(songs)
n_train = int(0.80 * n)
n_val   = int(0.10 * n)
train_songs = songs[:n_train]
val_songs   = songs[n_train:n_train+n_val]
test_songs  = songs[n_train+n_val:]

def tokenize_song_list(song_list):
    tokens = []
    for s in song_list:
        tokens.extend(s.split())
    return tokens

train_tokens = tokenize_song_list(train_songs)
val_tokens   = tokenize_song_list(val_songs)
test_tokens  = tokenize_song_list(test_songs)

print("Train tokens:", len(train_tokens))
print("Val tokens:", len(val_tokens))
# =========================
# 3) Build vocab ONLY from train (avoid leakage)
# =========================
with open("vocab.json", "r", encoding="utf-8") as f:
    token_to_id = json.load(f)

# Ensure PAD and UNK exist
PAD = "<PAD>"
UNK = "<UNK>"
if PAD not in token_to_id:
    token_to_id[PAD] = len(token_to_id)
if UNK not in token_to_id:
    token_to_id[UNK] = len(token_to_id)

id_to_token = {i: t for t, i in token_to_id.items()}
VOCAB_SIZE = len(token_to_id)

def encode(tokens):
    unk_id = token_to_id[UNK]
    return np.array([token_to_id.get(t, unk_id) for t in tokens], dtype=np.int32) #Why?

train_ids = encode(train_tokens)
val_ids   = encode(val_tokens)
test_ids  = encode(test_tokens)

print("Vocab size:", VOCAB_SIZE)

# =========================
# 4) Make (X, y) sequences for next-token prediction
# =========================
SEQ_LEN = 24  # context length (tune: 16..64)
BATCH_SIZE = 128 #Check if I am sure about this

def make_xy(ids: np.ndarray, seq_len: int):
    # sliding window
    X = []
    y = []
    for i in range(0, len(ids) - seq_len):
        X.append(ids[i:i+seq_len])
        y.append(ids[i+seq_len])
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)

X_train, y_train = make_xy(train_ids, SEQ_LEN)
X_val,   y_val   = make_xy(val_ids, SEQ_LEN)
X_test,  y_test  = make_xy(test_ids, SEQ_LEN)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape,   "y_val  :", y_val.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)

# tf.data for speed
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(20000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =========================
# 5) Model: Embedding -> LSTM -> Softmax over vocab
# =========================
EMB_DIM = 64
LSTM_UNITS = 128
LR = 1e-3

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMB_DIM, input_length=SEQ_LEN),
    tf.keras.layers.LSTM(LSTM_UNITS),
    tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax")
]) #The most important function creates sequential model

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

model.summary()

# Early stopping to avoid overfitting hard on 90 songs
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
]

# =========================
# 6) Train
# =========================
EPOCHS = 30
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# =========================
# 7) Test + Perplexity
# =========================
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
print(f"Test perplexity: {math.exp(test_loss):.2f}")
model.save("LSTM_model.keras")
# =========================
# 8) Generation
# =========================
def sample_from_probs(probs, temperature=1.0):
    probs = np.asarray(probs).astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(probs))
    logits = np.log(probs + 1e-12) / temperature
    exp = np.exp(logits - np.max(logits))
    p = exp / exp.sum()
    return int(np.random.choice(len(p), p=p))

def generate(start_tokens, steps=64, temperature=0.9):
    seq = [token_to_id.get(t, token_to_id[UNK]) for t in start_tokens]
    if len(seq) < SEQ_LEN:
        seq = [token_to_id[PAD]] * (SEQ_LEN - len(seq)) + seq

    generated = list(start_tokens)

    for _ in range(steps):
        x = np.array([seq[-SEQ_LEN:]], dtype=np.int32)
        probs = model.predict(x, verbose=0)[0]
        nxt = sample_from_probs(probs, temperature=temperature)
        token = id_to_token[nxt]

        if token in SECTION_TOKENS and generated and generated[-1] == token:
            continue

        seq.append(nxt)
        generated.append(token)

    return generated

print("\nSample generation:")
print(" ".join(generate(["<CHORUS>", "Cm", "Gm", "<BAR>"], steps=40, temperature=0.9)))