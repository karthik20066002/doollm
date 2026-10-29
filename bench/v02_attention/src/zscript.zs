version "4.14"

const A_D = 768;
const A_QKV = 2304;
const A_HEADS = 12;
const A_DH = 64;
const A_T = 4;

class AttentionBench : StaticEventHandler
{
    Array<double> emb, lng, lnb, sqkv, bqkv, sproj, bproj;
    Array<int> wqkv, wproj;
    Array<double> xh, qkvAll, outA, yAll;
    Array<int> xq, q8, k8, v8, oq;
    Array<double> sqT, skT, svT;
    bool benchRan;
    int resumeTicks;

    override void WorldLoaded(WorldEvent e)
    {
        Console.Printf("WORLD_LOADED_OK");
        if (benchRan) return;
        int specLump = Wads.CheckNumForFullName("data/benchspec.txt");
        if (specLump < 0) return;

        double loadMs;
        if (!LoadAll(loadMs)) return;
        Console.Printf("LOAD weights_ms=%.3f", loadMs);

        uint qi, ai, pi;
        Forward(qi, ai, pi, false);

        double t0 = MSTimeF();
        Forward(qi, ai, pi, true);
        double est = MSTimeF() - t0;
        int iters = 50;
        if (est > 0.005) iters = int(0.25 / est);
        if (iters < 1) iters = 1;
        if (iters > 50) iters = 50;

        double t1 = MSTimeF();
        for (int i = 0; i < iters; i++)
            Forward(qi, ai, pi, true);
        double total = MSTimeF() - t1;
        double macs = double(A_T) * (double(A_D) * A_QKV + double(A_D) * A_D) * iters;
        Console.Printf("TIME iters=%d total_ms=%.3f ms_per_forward=%.4f macs_per_sec=%.1f",
            iters, total, total / iters, macs / (total / 1000.0));

        benchRan = true;
        resumeTicks = 35;
        Console.Printf("MATMUL_DONE");
    }

    override void WorldTick()
    {
        if (resumeTicks > 0 && --resumeTicks == 0)
            Console.Printf("TICK_AFTER_BENCH game_resumed_ok");
    }

    override void NetworkProcess(ConsoleEvent e)
    {
        if (e.Name ~== "attn")
        {
            double loadMs;
            if (!LoadAll(loadMs)) return;
            uint qi, ai, pi;
            Forward(qi, ai, pi, false);
        }
    }

    double Pow2(int e)
    {
        double p = 1.0;
        if (e >= 0)
            for (int i = 0; i < e; i++) p *= 2.0;
        else
            for (int i = 0; i < -e; i++) p *= 0.5;
        return p;
    }

    double LoadFx(string b, int idx)
    {
        idx *= 8;
        uint lo = b.ByteAt(idx) | (b.ByteAt(idx + 1) << 8) |
                  (b.ByteAt(idx + 2) << 16) | (b.ByteAt(idx + 3) << 24);
        uint hi = b.ByteAt(idx + 4) | (b.ByteAt(idx + 5) << 8) |
                  (b.ByteAt(idx + 6) << 16) | (b.ByteAt(idx + 7) << 24);
        double bits = double(lo) + double(hi) * 4294967296.0;

        bool neg = bits >= 9223372036854775808.0;
        if (neg) bits -= 9223372036854775808.0;

        double ef = floor(bits / 4503599627370496.0);
        double m = bits - ef * 4503599627370496.0;
        int ei = ef;

        double v;
        if (ei == 0)
            v = m * Pow2(-1074);
        else
            v = Pow2(ei - 1023) * (1.0 + m * (1.0 / 4503599627370496.0));
        return neg ? -v : v;
    }

    bool LoadAll(out double ms)
    {
        double t0 = MSTimeF();
        int le = Wads.CheckNumForFullName("data/emb.bin");
        int lg = Wads.CheckNumForFullName("data/lng.bin");
        int lb = Wads.CheckNumForFullName("data/lnb.bin");
        int wq = Wads.CheckNumForFullName("data/wqkv.bin");
        int sq = Wads.CheckNumForFullName("data/sqkv.bin");
        int bq = Wads.CheckNumForFullName("data/bqkv.bin");
        int wp = Wads.CheckNumForFullName("data/wproj.bin");
        int sp = Wads.CheckNumForFullName("data/sproj.bin");
        int bp = Wads.CheckNumForFullName("data/bproj.bin");
        if (le < 0 || lg < 0 || lb < 0 || wq < 0 || sq < 0 || bq < 0 || wp < 0 || sp < 0 || bp < 0)
        {
            Console.Printf("MISSING_LUMPS");
            ms = 0;
            return false;
        }

        string embB = Wads.ReadLump(le);
        string lngB = Wads.ReadLump(lg);
        string lnbB = Wads.ReadLump(lb);
        string wqB = Wads.ReadLump(wq);
        string sqB = Wads.ReadLump(sq);
        string bqB = Wads.ReadLump(bq);
        string wpB = Wads.ReadLump(wp);
        string spB = Wads.ReadLump(sp);
        string bpB = Wads.ReadLump(bp);

        emb.Resize(A_T * A_D);
        for (int i = 0; i < A_T * A_D; i++) emb[i] = LoadFx(embB, i);
        lng.Resize(A_D);
        for (int i = 0; i < A_D; i++) lng[i] = LoadFx(lngB, i);
        lnb.Resize(A_D);
        for (int i = 0; i < A_D; i++) lnb[i] = LoadFx(lnbB, i);

        wqkv.Resize(A_QKV * A_D);
        for (int i = 0; i < A_QKV * A_D; i++) wqkv[i] = wqB.ByteAt(i);
        sqkv.Resize(A_QKV);
        for (int i = 0; i < A_QKV; i++) sqkv[i] = LoadFx(sqB, i);
        bqkv.Resize(A_QKV);
        for (int i = 0; i < A_QKV; i++) bqkv[i] = LoadFx(bqB, i);

        wproj.Resize(A_D * A_D);
        for (int i = 0; i < A_D * A_D; i++) wproj[i] = wpB.ByteAt(i);
        sproj.Resize(A_D);
        for (int i = 0; i < A_D; i++) sproj[i] = LoadFx(spB, i);
        bproj.Resize(A_D);
        for (int i = 0; i < A_D; i++) bproj[i] = LoadFx(bpB, i);

        xh.Resize(A_D);
        xq.Resize(A_D);
        qkvAll.Resize(A_T * A_QKV);
        q8.Resize(A_T * A_D);
        k8.Resize(A_T * A_D);
        v8.Resize(A_T * A_D);
        sqT.Resize(A_T * A_HEADS);
        skT.Resize(A_T * A_HEADS);
        svT.Resize(A_T * A_HEADS);
        outA.Resize(A_T * A_D);
        oq.Resize(A_D);
        yAll.Resize(A_T * A_D);

        ms = MSTimeF() - t0;
        return true;
    }

    int RintEven(double v)
    {
        double fl = floor(v);
        double fr = v - fl;
        int fi = fl;
        if (fr < 0.5) return fi;
        if (fr > 0.5) return fi + 1;
        int m = fi - (fi / 2) * 2;
        if (m < 0) m = -m;
        return m == 0 ? fi : fi + 1;
    }

    void QuantSlice(Array<double> src, int soff, Array<int> dst, int doff, int n, out double s)
    {
        double m = 0;
        for (int i = 0; i < n; i++)
        {
            double a = src[soff + i];
            if (a < 0) a = -a;
            if (a > m) m = a;
        }
        s = m / 127.0;
        if (s <= 0) s = 1.0;
        for (int i = 0; i < n; i++)
            dst[doff + i] = RintEven(src[soff + i] / s);
    }

    void LnQuantToken(int t)
    {
        double mean = 0;
        for (int i = 0; i < A_D; i++) mean += emb[t * A_D + i];
        mean /= A_D;

        double v = 0;
        for (int i = 0; i < A_D; i++)
        {
            double d = emb[t * A_D + i] - mean;
            v += d * d;
            xh[i] = d;
        }
        v /= A_D;

        double inv = 1.0 / sqrt(v + 0.00001);
        double mmax = 0;
        for (int i = 0; i < A_D; i++)
        {
            double h = xh[i] * inv * lng[i] + lnb[i];
            xh[i] = h;
            double a = h;
            if (a < 0) a = -a;
            if (a > mmax) mmax = a;
        }
        double sx = mmax / 127.0;
        if (sx <= 0) sx = 1.0;
        for (int i = 0; i < A_D; i++)
            xq[i] = RintEven(xh[i] / sx);
    }

    void MatvecQ(Array<int> w, Array<double> sc, Array<double> bi,
                 Array<int> qin, int nin, int nout,
                 Array<double> y, out uint isum)
    {
        int sumx = 0;
        for (int j = 0; j < nin; j++) sumx += qin[j];
        isum = 0;
        for (int o = 0; o < nout; o++)
        {
            int accb = 0;
            int base = o * nin;
            for (int j = 0; j < nin; j++)
                accb += w[base + j] * qin[j];
            int acc = accb - 128 * sumx;
            isum += acc;
            y[o] = sc[o] * acc + bi[o];
        }
    }

    void Forward(out uint qkvInt, out uint attInt, out uint projInt, bool quiet)
    {
        qkvInt = 0;
        attInt = 0;
        projInt = 0;

        for (int t = 0; t < A_T; t++)
        {
            LnQuantToken(t);
            uint xs = 0;
            for (int i = 0; i < A_D; i++) xs += xq[i];
            Array<double> yq;
            yq.Resize(A_QKV);
            uint part;
            MatvecQ(wqkv, sqkv, bqkv, xq, A_D, A_QKV, yq, part);
            qkvInt += part;
            for (int o = 0; o < A_QKV; o++) qkvAll[t * A_QKV + o] = yq[o];
            if (!quiet)
            {
                Console.Printf("XQ %d %.0f", t, double(xs));
                if (t == 0)
                    for (int i = 0; i < 8; i++)
                        Console.Printf("LNO %d %.17g", i, xh[i]);
            }
        }
        if (!quiet) Console.Printf("QKVINT - %.0f", double(qkvInt));

        for (int t = 0; t < A_T; t++)
            for (int h = 0; h < A_HEADS; h++)
            {
                double s;
                QuantSlice(qkvAll, t * A_QKV + h * A_DH, q8, t * A_D + h * A_DH, A_DH, s);
                sqT[t * A_HEADS + h] = s;
                QuantSlice(qkvAll, t * A_QKV + A_DH * A_HEADS + h * A_DH, k8, t * A_D + h * A_DH, A_DH, s);
                skT[t * A_HEADS + h] = s;
                QuantSlice(qkvAll, t * A_QKV + 2 * A_DH * A_HEADS + h * A_DH, v8, t * A_D + h * A_DH, A_DH, s);
                svT[t * A_HEADS + h] = s;
            }

        for (int h = 0; h < A_HEADS; h++)
            for (int t = 0; t < A_T; t++)
            {
                double s[A_T];
                int qbase = t * A_D + h * A_DH;
                for (int u = 0; u < A_T; u++)
                {
                    int qd = 0;
                    int kbase = u * A_D + h * A_DH;
                    for (int i = 0; i < A_DH; i++)
                        qd += q8[qbase + i] * k8[kbase + i];
                    attInt += qd;
                    s[u] = qd * 0.125 * sqT[t * A_HEADS + h] * skT[u * A_HEADS + h];
                }
                double m = s[0];
                for (int u = 1; u < A_T; u++) if (s[u] > m) m = s[u];
                double e[A_T];
                double esum = 0;
                for (int u = 0; u < A_T; u++)
                {
                    e[u] = exp(s[u] - m);
                    esum += e[u];
                }
                double p[A_T];
                for (int u = 0; u < A_T; u++) p[u] = e[u] / esum;
                if (!quiet && t == 0 && h == 0)
                    for (int u = 0; u < A_T; u++)
                        Console.Printf("P %d %.17g", u, p[u]);

                for (int j = 0; j < A_DH; j++)
                {
                    double acc = 0;
                    for (int u = 0; u < A_T; u++)
                        acc += p[u] * (svT[u * A_HEADS + h] * v8[u * A_D + h * A_DH + j]);
                    outA[t * A_D + h * A_DH + j] = acc;
                }
            }
        if (!quiet) Console.Printf("ATTINT - %.0f", double(attInt));

        for (int t = 0; t < A_T; t++)
        {
            double so;
            QuantSlice(outA, t * A_D, oq, 0, A_D, so);
            Array<double> y;
            y.Resize(A_D);
            uint part;
            MatvecQ(wproj, sproj, bproj, oq, A_D, A_D, y, part);
            projInt += part;
            for (int i = 0; i < A_D; i++) yAll[t * A_D + i] = y[i];
        }
        if (!quiet)
        {
            Console.Printf("PROJINT - %.0f", double(projInt));
            for (int i = 0; i < 16; i++)
                Console.Printf("Y %d %.17g", i, yAll[i]);
            double total = 0;
            for (int t = 0; t < A_T; t++)
                for (int i = 0; i < A_D; i++)
                    total += yAll[t * A_D + i];
            Console.Printf("YSUM - %.17g", total);
        }
    }
}
