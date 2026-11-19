/*
 * C core of data_prep/census/fast_tokenize.py: a port of
 *   mido 1.3.3 MidiFile parsing  +  pretty_midi 0.2.11 PrettyMIDI.__init__ (tempo map, metadata checks, note pairing)
 *   +  musicar.data.loader tokenize_tracks / tokenize_midi (sort, quantization)
 * that reproduces their results bit for bit, including which files are "broken" (any exception -> None).
 *
 * Build: gcc -O2 -fPIC -shared -ffp-contract=off (no FMA: numpy computes last_end + scale * ticks as two roundings).
 *
 * ft_tokenize() returns n >= 1 notes written to tokens/programs, 0 if the file parses but has no notes (-> None),
 * FT_BROKEN if mido or pretty_midi would raise (-> None), FT_FALLBACK if this code can't decide (out of memory or
 * the output buffer is too small) -> the caller runs the original tokenizer.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FT_BROKEN (-1)
#define FT_FALLBACK (-2)
#define MAX_TICK 10000000LL      /* pretty_midi: max_tick = max(abs time) + 1 > 1e7 -> ValueError */
#define MAX_MSG 1000000LL        /* mido MAX_MESSAGE_LENGTH */
#define SAT (1LL << 40)          /* varints / tick sums saturate here; every test above is far below it */

typedef struct { int32_t *tick; uint8_t *vel; int32_t len, cap; } openlist;

typedef struct {
    int32_t *st, *en, *inst; uint8_t *pitch, *vel, *prog;
    int64_t n, cap;
    int64_t *seg_tick; double *seg_scale; int64_t nseg, segcap;
    openlist open[16 * 128];
} ctx;

typedef struct { double st; uint32_t k2; uint32_t pos; uint32_t idx; } skey;

static int grow_notes(ctx *c) {
    int64_t cap = c->cap ? c->cap * 2 : 1024;
    void *p;
#define G(f, T) p = realloc(c->f, (size_t)cap * sizeof(T)); if (!p) return 0; c->f = p;
    G(st, int32_t) G(en, int32_t) G(inst, int32_t) G(pitch, uint8_t) G(vel, uint8_t) G(prog, uint8_t)
#undef G
    c->cap = cap;
    return 1;
}

static int push_seg(ctx *c, int64_t tick, double scale) {
    if (c->nseg == c->segcap) {
        int64_t cap = c->segcap ? c->segcap * 2 : 16;
        void *p = realloc(c->seg_tick, (size_t)cap * sizeof(int64_t)); if (!p) return 0; c->seg_tick = p;
        p = realloc(c->seg_scale, (size_t)cap * sizeof(double)); if (!p) return 0; c->seg_scale = p;
        c->segcap = cap;
    }
    c->seg_tick[c->nseg] = tick; c->seg_scale[c->nseg] = scale; c->nseg++;
    return 1;
}

static void free_ctx(ctx *c) {
    free(c->st); free(c->en); free(c->inst); free(c->pitch); free(c->vel); free(c->prog);
    free(c->seg_tick); free(c->seg_scale);
    for (int i = 0; i < 16 * 128; i++) { free(c->open[i].tick); free(c->open[i].vel); }
}

/* pretty_midi __tick_to_time[tick] for the segments so far (segment start values recomputed the numpy way). */
static double tick_time(const ctx *c, const double *seg_start, int64_t tick) {
    int64_t lo = 0, hi = c->nseg - 1;           /* last segment with seg_tick <= tick */
    while (lo < hi) { int64_t mid = (lo + hi + 1) / 2; if (c->seg_tick[mid] <= tick) lo = mid; else hi = mid - 1; }
    return seg_start[lo] + c->seg_scale[lo] * (double)(tick - c->seg_tick[lo]);
}

static double *seg_starts(const ctx *c) {
    double *s = malloc((size_t)c->nseg * sizeof(double));
    if (!s) return NULL;
    s[0] = 0.0;
    for (int64_t i = 1; i < c->nseg; i++)
        s[i] = s[i - 1] + c->seg_scale[i - 1] * (double)(c->seg_tick[i] - c->seg_tick[i - 1]);
    return s;
}

/* time of a key/time signature in track 0 must be >= 0 (pretty_midi containers); only fails with ticks_per_beat < 0 */
static int sig_time_ok(const ctx *c, int64_t tick) {
    double *s = seg_starts(c);
    if (!s) return -1;
    double t = tick_time(c, s, tick);
    free(s);
    return t >= 0.0;
}

static int cmp_skey(const void *a, const void *b) {
    const skey *x = a, *y = b;
    if (x->st < y->st) return -1;
    if (x->st > y->st) return 1;
    if (x->k2 != y->k2) return x->k2 < y->k2 ? -1 : 1;
    return x->pos < y->pos ? -1 : (x->pos > y->pos);
}

static int64_t qtime(double x, double res, int64_t max_bin) {
    double r = rint(x / res);                   /* Python round(): half to even, = rint in the default FP mode */
    if (r < 0) r = 0;
    if (r > (double)max_bin) r = (double)max_bin;
    return (int64_t)r;
}

#define RB(v) do { if (pos >= n) { rc = FT_BROKEN; goto done; } (v) = d[pos++]; } while (0)
#define RVAR(v) do { int64_t _x = 0; int _b; for (;;) { RB(_b); _x = (_x << 7) | (_b & 0x7f); if (_x > SAT) _x = SAT; \
                     if (_b < 0x80) break; } (v) = _x; } while (0)
#define BROKE do { rc = FT_BROKEN; goto done; } while (0)

/*
 * mode 0: tokenize_tracks (all notes, programs, sort by onset, program, pitch)
 * mode 1: tokenize_midi   (non-drum notes, sort by onset, pitch)
 * bad_den[k] = 1 when mido rejects time-signature denominator 2**k (math.log(2**k, 2) is not an integer)
 */
int64_t ft_tokenize(const uint8_t *d, int64_t n, int mode, const uint8_t *bad_den,
                    int velocity_bins, double time_res, int64_t max_dur, int64_t max_delta,
                    uint16_t *tokens, uint8_t *programs, int64_t out_cap) {
    ctx *c = calloc(1, sizeof(ctx));
    if (!c) return FT_FALLBACK;
    int64_t rc = 0, pos = 0;
    double *ss = NULL; skey *keys = NULL; int32_t *count = NULL;
    int32_t ninst = 0;
    int32_t imap[128][16];

    /* header: read_file_header */
    if (n < 8 || memcmp(d, "MThd", 4) != 0) BROKE;
    {
        int64_t hsize = ((int64_t)d[4] << 24) | (d[5] << 16) | (d[6] << 8) | d[7];
        int64_t avail = n - 8 < hsize ? n - 8 : hsize;
        if (avail < 6) BROKE;
        pos = 8 + avail;
    }
    int ntracks = (int16_t)((d[10] << 8) | d[11]);
    int res = (int16_t)((d[12] << 8) | d[13]);
    if (ntracks <= 0) BROKE;                         /* no tracks -> max([]) in pretty_midi */
    if (res == 0) BROKE;                             /* 60.0 / (120.0 * 0) */
    /* ticks_per_beat < 0 (SMPTE-style division): pretty_midi uses it as is, so time runs backwards and Note() raises
       when end < start. Rare; leave those files to the original tokenizer. */
    if (res < 0) { rc = FT_FALLBACK; goto done; }
    if (!push_seg(c, 0, 60.0 / (120.0 * (double)res))) { rc = FT_FALLBACK; goto done; }

    for (int track = 0; track < ntracks; track++) {
        if (n - pos < 8) BROKE;
        if (memcmp(d + pos, "MTrk", 4) != 0) BROKE;
        int64_t size = ((int64_t)d[pos + 4] << 24) | (d[pos + 5] << 16) | (d[pos + 6] << 8) | d[pos + 7];
        pos += 8;
        int64_t start = pos, tick = 0, nev = 0;
        int last_status = -1;
        int prog[16] = {0};
        memset(imap, 0xff, sizeof(imap));
        for (int i = 0; i < 16 * 128; i++) c->open[i].len = 0;   /* notes left open at the end of a track are dropped */

        for (;;) {
            if (pos - start == size) break;
            if (pos - start > size) BROKE;           /* mido never stops after overshooting the chunk -> EOFError */
            int64_t delta; int status, peek = -1;
            RVAR(delta);
            int64_t prev_tick = tick;
            tick += delta; if (tick > SAT) tick = SAT;
            RB(status);
            if (status < 0x80) {
                if (last_status < 0) BROKE;          /* running status without last_status */
                peek = status; status = last_status;
            } else if (status != 0xff) {
                last_status = status;                /* also sysex and system messages, as mido does */
            }
            nev++;

            if (status == 0xff) {                    /* meta */
                int type; int64_t len;
                RB(type); RVAR(len);
                /* mido builds UnknownMetaMessage(type, data) without the delta: its time is 0 (delta dropped) */
                if (!((type >= 0x00 && type <= 0x07) || type == 0x09 || type == 0x20 || type == 0x21 ||
                      type == 0x2f || type == 0x51 || type == 0x54 || type == 0x58 || type == 0x59 || type == 0x7f))
                    tick = prev_tick;
                if (tick >= MAX_TICK) BROKE;
                if (len > MAX_MSG || n - pos < len) BROKE;
                const uint8_t *m = d + pos; pos += len;
                switch (type) {
                case 0x00: if (len == 1) BROKE; break;
                case 0x20: if (len == 0) BROKE; break;
                case 0x51: {
                    if (len < 3) BROKE;
                    int64_t tempo = ((int64_t)m[0] << 16) | (m[1] << 8) | m[2];
                    if (track == 0) {
                        if (tempo == 0) BROKE;
                        double scale = 60.0 / ((6e7 / (double)tempo) * (double)res);
                        if (tick == 0) { c->nseg = 0; if (!push_seg(c, 0, scale)) { rc = FT_FALLBACK; goto done; } }
                        else if (scale != c->seg_scale[c->nseg - 1] && !push_seg(c, tick, scale)) {
                            rc = FT_FALLBACK; goto done; }
                    }
                    break; }
                case 0x54:
                    if (len < 1 || (m[0] >> 5) > 3) BROKE;
                    if (len < 5 || m[1] > 59 || m[2] > 59 || m[4] > 99) BROKE;
                    break;
                case 0x58:
                    if (len < 4 || bad_den[m[1]]) BROKE;
                    if (track == 0) {
                        if (m[0] == 0) BROKE;        /* pretty_midi TimeSignature: numerator > 0 */
                        int ok = sig_time_ok(c, tick); if (ok < 0) { rc = FT_FALLBACK; goto done; } if (!ok) BROKE;
                    }
                    break;
                case 0x59: {
                    if (len < 2) BROKE;
                    int key = (int8_t)m[0];
                    if (key < -7 || key > 7 || m[1] > 1) BROKE;   /* KeySignatureError */
                    if (track == 0) {
                        int ok = sig_time_ok(c, tick); if (ok < 0) { rc = FT_FALLBACK; goto done; } if (!ok) BROKE;
                    }
                    break; }
                default: break;
                }
            } else if (status == 0xf0 || status == 0xf7) {   /* sysex; a running-status peek byte is discarded */
                int64_t len;
                if (tick >= MAX_TICK) BROKE;
                RVAR(len);
                if (len > MAX_MSG || n - pos < len) BROKE;
                const uint8_t *m = d + pos; pos += len;
                int64_t a = 0, b = len;
                if (b > a && m[a] == 0xf0) a++;
                if (b > a && m[b - 1] == 0xf7) b--;
                for (int64_t i = a; i < b; i++) if (m[i] > 127) BROKE;
            } else {
                int length;
                if (tick >= MAX_TICK) BROKE;
                if (status < 0xf0) length = ((status & 0xf0) == 0xc0 || (status & 0xf0) == 0xd0) ? 2 : 3;
                else switch (status) {
                    case 0xf1: case 0xf3: length = 2; break;
                    case 0xf2: length = 3; break;
                    case 0xf6: case 0xf8: case 0xfa: case 0xfb: case 0xfc: case 0xfe: length = 1; break;
                    default: BROKE;                  /* undefined status byte */
                }
                int have = peek >= 0, size_rd = length - 1 - have;
                if (size_rd < 0) BROKE;              /* wrong number of bytes */
                int db[2]; int k = 0;
                if (have) db[k++] = peek;
                for (int i = 0; i < size_rd; i++) { int b; RB(b); db[k++] = b; }
                for (int i = have; i < k; i++) if (db[i] > 127) BROKE;
                if (status >= 0xf0) continue;
                int hi = status & 0xf0, ch = status & 0x0f;
                if (hi == 0xc0) { prog[ch] = db[0]; }
                else if (hi == 0x90 && db[1] > 0) {
                    openlist *L = &c->open[ch * 128 + db[0]];
                    if (L->len == L->cap) {
                        int32_t cap = L->cap ? L->cap * 2 : 4;
                        void *p = realloc(L->tick, (size_t)cap * sizeof(int32_t)); if (!p) { rc = FT_FALLBACK; goto done; }
                        L->tick = p;
                        p = realloc(L->vel, (size_t)cap); if (!p) { rc = FT_FALLBACK; goto done; }
                        L->vel = p; L->cap = cap;
                    }
                    L->tick[L->len] = (int32_t)tick; L->vel[L->len] = (uint8_t)db[1]; L->len++;
                } else if (hi == 0x80 || hi == 0x90) {
                    openlist *L = &c->open[ch * 128 + db[0]];
                    if (L->len == 0) continue;       /* spurious note-off */
                    int32_t keep = 0;
                    while (keep < L->len && L->tick[L->len - 1 - keep] == tick) keep++;
                    int32_t nclose = L->len - keep;
                    for (int32_t i = 0; i < nclose; i++) {
                        int p = prog[ch];
                        int32_t id = imap[p][ch];
                        if (id < 0) id = imap[p][ch] = ninst++;   /* (program at note-off, channel, track) */
                        if (mode == 1 && ch == 9) continue;
                        if (c->n == c->cap && !grow_notes(c)) { rc = FT_FALLBACK; goto done; }
                        int64_t j = c->n++;
                        c->st[j] = L->tick[i]; c->en[j] = (int32_t)tick; c->inst[j] = id;
                        c->pitch[j] = (uint8_t)db[0]; c->vel[j] = L->vel[i]; c->prog[j] = ch == 9 ? 128 : (uint8_t)p;
                    }
                    if (nclose > 0 && keep > 0) {
                        memmove(L->tick, L->tick + nclose, (size_t)keep * sizeof(int32_t));
                        memmove(L->vel, L->vel + nclose, (size_t)keep);
                        L->len = keep;
                    } else {
                        L->len = 0;                  /* incl. the note-on at this very tick when nothing closed */
                    }
                }
            }
        }
        if (nev == 0) BROKE;                         /* empty track -> max([]) in pretty_midi */
    }

    if (c->n == 0) { rc = 0; goto done; }
    if (c->n > out_cap || c->n >= (1LL << 31)) { rc = FT_FALLBACK; goto done; }

    /* flattened position = order in [n for inst in instruments for n in inst.notes] (stable-sort tiebreak) */
    count = calloc((size_t)ninst + 1, sizeof(int32_t));
    keys = malloc((size_t)c->n * sizeof(skey));
    ss = seg_starts(c);
    if (!count || !keys || !ss) { rc = FT_FALLBACK; goto done; }
    for (int64_t j = 0; j < c->n; j++) count[c->inst[j] + 1]++;
    for (int32_t i = 0; i < ninst; i++) count[i + 1] += count[i];
    for (int64_t j = 0; j < c->n; j++) {
        keys[j].st = tick_time(c, ss, c->st[j]);
        keys[j].k2 = mode == 0 ? ((uint32_t)c->prog[j] << 8) | c->pitch[j] : c->pitch[j];
        keys[j].pos = (uint32_t)count[c->inst[j]]++;
        keys[j].idx = (uint32_t)j;
    }
    qsort(keys, (size_t)c->n, sizeof(skey), cmp_skey);
    double prev = keys[0].st;
    for (int64_t i = 0; i < c->n; i++) {
        uint32_t j = keys[i].idx;
        double st = keys[i].st, en = tick_time(c, ss, c->en[j]);
        double length = en - st, dt = st - prev;
        int v = c->vel[j] * velocity_bins / 128;
        if (v > velocity_bins - 1) v = velocity_bins - 1;
        if (v < 0) v = 0;
        tokens[4 * i + 0] = c->pitch[j];
        tokens[4 * i + 1] = (uint16_t)v;
        tokens[4 * i + 2] = (uint16_t)qtime(length > 0.0 ? length : 0.0, time_res, max_dur);
        tokens[4 * i + 3] = (uint16_t)qtime(dt > 0.0 ? dt : 0.0, time_res, max_delta);
        programs[i] = c->prog[j];
        prev = st;
    }
    rc = c->n;

done:
    free(ss); free(keys); free(count);
    free_ctx(c); free(c);
    return rc;
}
