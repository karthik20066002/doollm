#!/usr/bin/env python3


import argparse
import sys
import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "src" / "data"

D, INTER, L = 896, 4864, 24
HEADS, KVH, HD, GQA = 14, 2, 64, 7
EPS, ATTN_SCALE = 1e-6, 0.125
MAXTOK = 24
LAYERS_KEY = (0, 12, 23)

W8 = {}
SC = {}


def load(prompt_path=None):
    def u8(name):
        return np.fromfile(DATA / name, dtype=np.uint8).astype(np.int32)

    def f64(name):
        return np.fromfile(DATA / name, dtype="<f8")

    for k, wn, sn in [("q", "wq.bin", "sq.bin"), ("k", "wk.bin", "sk.bin"),
                      ("v", "wv.bin", "sv.bin"), ("o", "wo.bin", "so.bin"),
                      ("g", "wg.bin", "sg.bin"), ("u", "wu.bin", "su.bin"),
                      ("d", "wd.bin", "sd.bin")]:
        W8[k] = u8(wn).reshape(L, -1) - 128
        SC[k] = f64(sn).reshape(L, -1)
    W8["te"] = (u8("wte.bin") - 128).reshape(-1, D)
    SC["te"] = f64("swte.bin")
    BQ = f64("bq.bin").reshape(L, D)
    BK = f64("bk.bin").reshape(L, KVH * HD)
    BV = f64("bv.bin").reshape(L, KVH * HD)
    cfg = dict(n1=f64("norm1.bin").reshape(L, D), n2=f64("norm2.bin").reshape(L, D),
               nf=f64("normf.bin"), rope=f64("rope.bin"))
    pids = np.fromfile(prompt_path or DATA / "prompt.bin", dtype="<i4")
    if len(pids)<2 or pids[0] != len(pids)-1 or np.any(pids[1:]<0) or np.any(pids[1:]>=151936):
        raise ValueError("invalid prompt file")
    cfg["plen"] = int(pids[0])
    cfg["pids"] = pids[1:]
    cfg["cos"] = cfg["rope"][: 128 * 32].reshape(128, 32)
    cfg["sin"] = cfg["rope"][128 * 32:].reshape(128, 32)
    cfg["bq"] = BQ
    cfg["bk"] = BK
    cfg["bv"] = BV
    return cfg


def expd(x: float) -> float:

    if x < -700.0:
        return 0.0
    n = math.floor(x + 0.5)
    f = x - n
    p = 1.5619206968586225e-16
    p = p * f + 2.8114572543455206e-15
    p = p * f + 4.779477332387385e-14
    p = p * f + 7.647163731819816e-13
    p = p * f + 1.1470745597729725e-11
    p = p * f + 1.6059043836821613e-10
    p = p * f + 2.08767569878681e-09
    p = p * f + 2.505210838544172e-08
    p = p * f + 2.755731922398589e-07
    p = p * f + 2.7557319223985893e-06
    p = p * f + 2.48015873015873e-05
    p = p * f + 0.0001984126984126984
    p = p * f + 0.001388888888888889
    p = p * f + 0.008333333333333333
    p = p * f + 0.041666666666666664
    p = p * f + 0.16666666666666666
    p = p * f + 0.5
    p = p * f + 1.0
    p = p * f + 1.0
    nn = int(n)
    r = 1.0
    if nn >= 0:
        for _ in range(nn):
            r *= 2.718281828459045
    else:
        for _ in range(-nn):
            r *= 2.718281828459045
        r = 1.0 / r
    return r * p


def quant(v):
    s = float(np.abs(v).max()) / 127.0
    if s <= 0:
        s = 1.0
    return np.rint(v / s).astype(np.int32), s


def quant_per_row(v):


    s = np.abs(v).max(axis=1) / 127.0
    s[s == 0] = 1.0
    return np.rint(v / s[:, None]).astype(np.int32), s


def matvec(w8, sc, qin, bias, xs):
    acc = w8.reshape(-1, qin.shape[0]) @ qin
    return sc * acc * xs + bias, int(acc.sum()) & 0xFFFFFFFF


def rms(x, g):

    ms = 0.0
    for v in x:
        ms += float(v) * float(v)
    ms /= D
    inv = 1.0 / np.sqrt(ms + EPS)
    return x * inv * g


def rope(x, pos, cfg):
    c, s = cfg["cos"][pos], cfg["sin"][pos]
    a = x[:32].copy()
    b = x[32:].copy()
    x[:32] = a * c - b * s
    x[32:] = a * s + b * c
    return x


def esc(b: bytes) -> str:
    out = []
    for ch in b:
        if 32 <= ch < 127 and ch != 92:
            out.append(chr(ch))
        else:
            out.append("\\x%02X" % ch)
    return "".join(out)


def decode_table():
    raw = (DATA / "decode.bin").read_bytes()
    pos, table = 0, []
    while pos < len(raw):
        n = raw[pos]
        table.append(raw[pos + 1:pos + 1 + n])
        pos += 1 + n
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', type=Path)
    ap.add_argument('--max-tokens', type=int, default=MAXTOK, choices=range(1,MAXTOK+1))
    args = ap.parse_args()
    cfg = load(args.prompt)
    plen = cfg["plen"]
    print("PIDS", *map(int, cfg["pids"]))
    maxpos = plen + args.max_tokens - 1
    assert maxpos <= 128, "prompt too long for RoPE table"

    Kc = np.zeros((L, KVH, 128, HD))
    Vc = np.zeros((L, KVH, 128, HD))
    Skc = np.ones((L, KVH, 128))
    Svc = np.ones((L, KVH, 128))
    sums = {k: np.zeros(L, dtype=np.uint64) for k in ("q", "k", "v", "o", "g", "u", "d")}
    logit_acc = 0
    first_logits = None
    table = decode_table()

    for t in range(min(3, plen)):
        row = W8["te"][cfg["pids"][t]]
        print(f"ESUM {t} {int(row.sum()) & 0xFFFFFFFF}")

    def forward(pos, tid, need_logits):
        nonlocal logit_acc, first_logits
        ZO = np.zeros(D)
        ZI = np.zeros(INTER)
        x = W8["te"][tid].astype(np.float64) * SC["te"][tid]
        import os
        if pos == 0 and os.environ.get("DBG"):
            h0 = rms(x, cfg["n1"][0])
            xq0, sx10 = quant(h0)
            wq_sum = 0
            for _i in range(896):
                wq_sum = (wq_sum + (_i + 1) * int(xq0[_i])) & 0xFFFFFFFF
            print(f"FBB x0 {x[0]:.17g} h0 {h0[0]:.17g} sx1 {sx10:.17g} xq0 {xq0[0]} xq1 {xq0[1]} xq2 {xq0[2]} wq {wq_sum}")
        for l in range(L):
            h = rms(x, cfg["n1"][l])
            xq, sx1 = quant(h)
            qq, si = matvec(W8["q"][l], SC["q"][l], xq, cfg["bq"][l], sx1)
            sums["q"][l] += si
            kk, si = matvec(W8["k"][l], SC["k"][l], xq, cfg["bk"][l], sx1)
            sums["k"][l] += si
            vv, si = matvec(W8["v"][l], SC["v"][l], xq, cfg["bv"][l], sx1)
            sums["v"][l] += si

            q = qq.reshape(HEADS, HD).copy()
            k = kk.reshape(KVH, HD).copy()
            for hh in range(HEADS):
                rope(q[hh], pos, cfg)
            for hh in range(KVH):
                rope(k[hh], pos, cfg)
            kq, sk = quant_per_row(k)
            vq, sv = quant_per_row(vv.reshape(KVH, HD))
            Kc[l, :, pos] = kq
            Vc[l, :, pos] = vq
            Skc[l, :, pos] = sk
            Svc[l, :, pos] = sv


            n = pos + 1
            att = np.empty(D)
            for hh in range(HEADS):
                kvh = hh // GQA
                s = np.empty(n)
                for u in range(n):
                    ksu = Skc[l, kvh, u]
                    qd = 0.0
                    for j in range(HD):
                        qd += (ksu * Kc[l, kvh, u, j]) * q[hh, j]
                    s[u] = qd * ATTN_SCALE
                m = s.max()
                e = np.array([expd(float(v - m)) for v in s])
                es = 0.0
                for v in e:
                    es += float(v)
                for j in range(HD):
                    acc = 0.0
                    for u in range(n):
                        acc += (e[u] / es) * (Svc[l, kvh, u] * Vc[l, kvh, u, j])
                    att[hh * HD + j] = acc
            ao, sxo = quant(att)
            o, si = matvec(W8["o"][l], SC["o"][l], ao, ZO, sxo)
            sums["o"][l] += si
            x = x + o

            h2 = rms(x, cfg["n2"][l])
            xq2, sx2 = quant(h2)
            gg, si = matvec(W8["g"][l], SC["g"][l], xq2, ZI, sx2)
            sums["g"][l] += si
            uu, si = matvec(W8["u"][l], SC["u"][l], xq2, ZI, sx2)
            sums["u"][l] += si
            sil = np.empty(INTER)
            for _i in range(INTER):
                sil[_i] = gg[_i] / (1.0 + expd(-float(gg[_i]))) * uu[_i]
            mq, sxm = quant(sil)
            dd, si = matvec(W8["d"][l], SC["d"][l], mq, ZO, sxm)
            sums["d"][l] += si
            x = x + dd
            import os
            if pos == 0 and l == 0 and os.environ.get("DBG"):
                print(f"FB emb0 {x[0]:.17g}")
                print(f"FB sx1 {sx1:.17g}")
                print(f"FB emb00 {x[0]:.17g} h0 {h[0]:.17g} ms {float((x*x).sum()):.17g}")
                print(f"FB q0 {qq[0]:.17g} k0 {kk[0]:.17g} v0 {vv[0]:.17g}")
                print(f"FB att0 {att[0]:.17g} sxo {sxo:.17g} ao0 {ao[0]}")
                print(f"FB o0 {o[0]:.17g}")
                print(f"FB xatt0 {x[0]:.17g}")
                print(f"FB gg0 {gg[0]:.17g} uu0 {uu[0]:.17g} mq0 {mq[0]} sxm {sxm:.17g}")
                print(f"FB dd0 {dd[0]:.17g}")
                print(f"FB x0 {x[0]:.17g}")
            if pos <= 1 and os.environ.get("DBG"):
                print(f"DBG {pos} {l} {x[0]:.17g}")

        if need_logits:
            hf = rms(x, cfg["nf"])
            xqf, sxf = quant(hf)
            acc = W8["te"].reshape(-1, D) @ xqf
            logit_acc = int(acc.sum()) & 0xFFFFFFFF
            logits = SC["te"] * acc * sxf
            aid = int(np.argmax(logits))
            if first_logits is None:
                first_logits = (aid, float(logits[aid]))
            return aid
        return -1


    for p in range(plen):
        nxt = forward(p, cfg["pids"][p], p == plen - 1)
    print(f"LOGIT0 {first_logits[0]} {first_logits[1]:.17g} {logit_acc}")

    tid = nxt
    reply = b""
    for n in range(args.max_tokens):
        tb = table[tid] if tid < len(table) else b""
        if tid not in (151643, 151645):
            reply += tb
        print(f"TOK {n} {tid}")
        print(f"TTXT {n} {esc(tb)}")
        if tid in (151643, 151645) or n + 1 == args.max_tokens:
            break
        tid = forward(plen + n, tid, True)
    print(f"TEXT {esc(reply)}")

    for k in ("q", "k", "v", "o", "g", "u", "d"):
        for l in LAYERS_KEY:
            print(f"L{k.upper()} {l} {int(sums[k][l]) & 0xFFFFFFFF}")
        print(f"L{k.upper()} 99 {int(sums[k].sum()) & 0xFFFFFFFF}")


if __name__ == "__main__":
    main()
