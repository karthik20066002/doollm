#!/usr/bin/env python3


from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "src" / "data"
FX = 1 << 40
D, QKV, HEADS, DH, T = 768, 2304, 12, 64, 4
KSCALE = 1.0 / np.sqrt(DH)


def load_fx(name):
    return np.fromfile(DATA / name, dtype="<f8")


def quant(v):
    s = float(np.abs(v).max()) / 127.0
    if s <= 0:
        s = 1.0
    return np.rint(v / s).astype(np.int64), s


def main():
    emb = load_fx("emb.bin").reshape(T, D)
    lng, lnb = load_fx("lng.bin"), load_fx("lnb.bin")
    wqkv = np.fromfile(DATA / "wqkv.bin", dtype=np.uint8).reshape(QKV, D).astype(np.int64) - 128
    sqkv, bqkv = load_fx("sqkv.bin"), load_fx("bqkv.bin")
    wproj = np.fromfile(DATA / "wproj.bin", dtype=np.uint8).reshape(D, D).astype(np.int64) - 128
    sproj, bproj = load_fx("sproj.bin"), load_fx("bproj.bin")

    qkv_all = np.zeros((T, QKV))
    xq_sums = []
    qkv_int = 0
    for t in range(T):
        x = emb[t]

        mean = 0.0
        for v in x:
            mean += v
        mean /= D
        var = 0.0
        xc = np.empty(D)
        for i in range(D):
            xc[i] = x[i] - mean
            var += xc[i] * xc[i]
        var /= D
        inv = 1.0 / np.sqrt(var + 1e-5)
        xh = xc * inv * lng + lnb
        if t == 0:
            for i in range(8):
                print(f"LNO {i} {xh[i]:.17g}")
        xq, _sx = quant(xh)
        xq_sums.append(int(xq.sum()) % (1 << 32))
        acc = wqkv @ xq
        qkv_int = (qkv_int + int(acc.sum())) % (1 << 32)
        qkv_all[t] = sqkv * acc + bqkv

    q8 = np.zeros((T, D), dtype=np.int64)
    k8 = np.zeros((T, D), dtype=np.int64)
    v8 = np.zeros((T, D), dtype=np.int64)
    sq = np.zeros((T, HEADS))
    sk = np.zeros((T, HEADS))
    sv = np.zeros((T, HEADS))
    for t in range(T):
        for h in range(HEADS):
            q8[t, h * DH:(h + 1) * DH], sq[t, h] = quant(qkv_all[t, h * DH:(h + 1) * DH])
            k8[t, h * DH:(h + 1) * DH], sk[t, h] = quant(qkv_all[t, DH * HEADS + h * DH:DH * HEADS + (h + 1) * DH])
            v8[t, h * DH:(h + 1) * DH], sv[t, h] = quant(qkv_all[t, 2 * DH * HEADS + h * DH:2 * DH * HEADS + (h + 1) * DH])
    for t in range(T):
        print(f"XQ {t} {xq_sums[t]}")
    print(f"QKVINT - {qkv_int}")

    att_int = 0
    out = np.zeros((T, D))
    for h in range(HEADS):
        for t in range(T):
            s = np.empty(T)
            for u in range(T):
                qd = int((q8[t, h * DH:(h + 1) * DH] * k8[u, h * DH:(h + 1) * DH]).sum())
                att_int = (att_int + qd) % (1 << 32)
                s[u] = qd * KSCALE * sq[t, h] * sk[u, h]
            m = s.max()
            e = np.exp(s - m)
            p = e / e.sum()
            if t == 0 and h == 0:
                for u in range(T):
                    print(f"P {u} {p[u]:.17g}")
            for j in range(DH):
                acc = 0.0
                for u in range(T):
                    acc += p[u] * (sv[u, h] * v8[u, h * DH + j])
                out[t, h * DH + j] = acc
    print(f"ATTINT - {att_int}")

    proj_int = 0
    y = np.zeros((T, D))
    for t in range(T):
        oq, _so = quant(out[t])
        acc = wproj @ oq
        proj_int = (proj_int + int(acc.sum())) % (1 << 32)
        y[t] = sproj * acc + bproj
    print(f"PROJINT - {proj_int}")

    for i in range(16):
        print(f"Y {i} {y[0, i]:.17g}")
    total = 0.0
    for t in range(T):
        for i in range(D):
            total += y[t, i]
    print(f"YSUM - {total:.17g}")


if __name__ == "__main__":
    main()
