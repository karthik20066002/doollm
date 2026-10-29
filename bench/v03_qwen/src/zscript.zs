version "4.14"
#include "tokenizer.zs"

const Q_D = 896;
const Q_INTER = 4864;
const Q_L = 24;
const Q_HEADS = 14;
const Q_KVH = 2;
const Q_HD = 64;
const Q_GQA = 7;
const Q_MAXPOS = 128;
const Q_MAXTOK = 24;
const Q_VOCAB = 151936;
const Q_EOS = 151645;
const Q_EPS = 0.000001;

class QwenBench : StaticEventHandler
{

    Array<uint8> wq, wk, wv, wo, wg, wu, wd;
    Array<uint8> wte;
    Array<double> sq, sk, sv, so, sg, su, sd;
    Array<double> ste;
    Array<double> n1, n2;
    Array<double> nf;
    Array<double> rope;
    Array<double> bq, bk, bv;
    Array<double> bo, bg, bu, bd;
    Array<double> bte;

    Array<int> kc, vc;
    Array<double> skc, svc;

    Array<double> x, h, qArr, kArr, vArr, sil, att, hf, logits, upv;
    Array<int> xq, ao, xq2, mq, xqf;
    Array<double> e, p;
    Array<int> kq8, vq8;
    Array<double> ksc, vsc;

    Array<uint> lq, lk, lv, lo, lg, lu, ld;

    Array<int> pids;
    Array<int> decOff;
    string decBlob;
    int plen;
    int logit0Id, nTok;
    double logit0Max;
    uint logit0Sum;
    bool logit0Done;
    bool weightsReady;
    bool loadFailed;
    int pendingTicks;
    QwenTokenizer tokenizer;
    int resumeTicks;

    override void WorldLoaded(WorldEvent e)
    {
        Console.Printf("Qwen ready: qwen_prompt \"your question\" then qwen. Clear qwen_prompt to use qwen_idx.");
        if (CVar.GetCVar('qwen_autorun').GetBool())
        { pendingTicks = 2; CVar.GetCVar('qwen_autorun').SetBool(false); }
    }

    override void WorldTick()
    {
        if (pendingTicks > 0 && --pendingTicks == 0) RunBench();
        if (resumeTicks > 0 && --resumeTicks == 0)
            Console.Printf("TICK_AFTER_BENCH game_resumed_ok");
    }

    override void NetworkProcess(ConsoleEvent e)
    {
        if (e.Name ~== "qwen") RunBench();
        if (e.Name ~== "qwen_tokenize") LoadPrompt();
    }

    void RunBench()
    {
        Console.Printf("WORLD_LOADED_OK");
        if (!LoadPrompt()) return;
        if (CVar.GetCVar('qwen_tokenize_only').GetBool())
        { Console.Printf("TOKENIZE_DONE"); return; }
        int specLump = Wads.CheckNumForFullName("data/benchspec.txt");
        if (specLump < 0) return;

        double loadMs;
        if (!LoadAll(loadMs)) return;
        Console.Printf("LOAD weights_ms=%.3f", loadMs);
        logit0Done = false; logit0Id = 0; logit0Max = 0; logit0Sum = 0;
        for (int l=0; l<Q_L; l++)
        { lq[l]=0; lk[l]=0; lv[l]=0; lo[l]=0; lg[l]=0; lu[l]=0; ld[l]=0; }
        int maxTokens = clamp(CVar.GetCVar('qwen_max_tokens').GetInt(),1,Q_MAXTOK);

        for (int t = 0; t < 3 && t < plen; t++)
        {
            uint s = 0;
            int base = pids[t] * Q_D;
            for (int j = 0; j < Q_D; j++)
            {
                int wb = wte[base + j];
                s += uint(wb - 128);
            }
            Console.Printf("ESUM %d %.0f", t, double(s));
        }

        int nextId = -1;
        for (int pos = 0; pos < plen; pos++)
        {
            nextId = ForwardPos(pos, pids[pos], pos == plen - 1);
            Console.Printf("POS %d done", pos);
        }
        Console.Printf("LOGIT0 %d %.17g %.0f", logit0Id, logit0Max, double(logit0Sum));

        string text;
        double genMs = 0;
        nTok = 0;
        int tid = nextId;
        for (int n = 0; n < maxTokens; n++)
        {
            string tb = Detok(tid);
            Console.Printf("TOK %d %d", n, tid);
            Console.Printf("TTXT %d %s", n, EscapeBytes(tb));
            if (tid != Q_EOS && tid != 151643) text = text .. tb;
            nTok++;
            if (tid == Q_EOS || tid == 151643 || n+1 == maxTokens) break;
            double t0 = MSTimeF();
            tid = ForwardPos(plen + n, tid, true);
            genMs += MSTimeF() - t0;
        }
        Console.Printf("TEXT %s", EscapeBytes(text));
        Console.Printf("Qwen: %s", text);
        Console.Printf("GEN tokens=%d gen_ms=%.1f ms_per_token=%.3f",
            nTok, genMs, genMs / max(nTok - 1, 1));

        PrintSums("LQ", lq);
        PrintSums("LK", lk);
        PrintSums("LV", lv);
        PrintSums("LO", lo);
        PrintSums("LG", lg);
        PrintSums("LU", lu);
        PrintSums("LD", ld);

        resumeTicks = 35;
        Console.Printf("MATMUL_DONE");
    }

    double LoadFx(string b, int idx)
    {
        idx *= 8;
        uint lo = b.ByteAt(idx) | (b.ByteAt(idx + 1) << 8) |
                  (b.ByteAt(idx + 2) << 16) | (b.ByteAt(idx + 3) << 24);
        uint hi = b.ByteAt(idx + 4) | (b.ByteAt(idx + 5) << 8) |
                  (b.ByteAt(idx + 6) << 16) | (b.ByteAt(idx + 7) << 24);

        bool neg = (hi & 2147483648u) != 0;
        int ei = int((hi >> 20) & 2047u);
        double m52 = double(hi & 1048575u) * 4294967296.0 + double(lo);
        double v;
        if (ei == 0) v = m52 * Pow2(-1074);
        else v = (1.0 + m52 * (1.0 / 4503599627370496.0)) * Pow2(ei - 1023);
        return neg ? -v : v;
    }

    double Pow2(int ex)
    {
        double r = 1.0;
        if (ex >= 0) for (int i = 0; i < ex; i++) r *= 2.0;
        else for (int i = 0; i < -ex; i++) r *= 0.5;
        return r;
    }

    int LoadI32(string b, int idx)
    {
        idx *= 4;
        return b.ByteAt(idx) | (b.ByteAt(idx + 1) << 8) |
               (b.ByteAt(idx + 2) << 16) | (b.ByteAt(idx + 3) << 24);
    }

    bool LoadPrompt()
    {
        string query=CVar.GetCVar('qwen_prompt').GetString();
        int maxTokens=clamp(CVar.GetCVar('qwen_max_tokens').GetInt(),1,Q_MAXTOK);
        if (query.Length()>0)
        {
            if (tokenizer==null) tokenizer=new("QwenTokenizer");
            string system=CVar.GetCVar('qwen_system').GetString();
            string prompt="<|im_start|>system\n"..system.."<|im_end|>\n<|im_start|>user\n"..query.."<|im_end|>\n<|im_start|>assistant\n";
            if (!tokenizer.Encode(prompt,pids)) return false;
            plen=pids.Size();
        }
        else
        {
            int idx=CVar.GetCVar('qwen_idx').GetInt();
            string name=String.Format("data/prompt_%d.bin",idx);
            int lump=Wads.CheckNumForFullName(name);
            if (lump<0) { Console.Printf("QWEN_ERROR missing prompt slot %d",idx); return false; }
            string blob=Wads.ReadLump(lump);
            if (blob.Length()<4) { Console.Printf("QWEN_ERROR truncated prompt"); return false; }
            plen=LoadI32(blob,0);
            if (plen<1 || plen>Q_MAXPOS || blob.Length()!=4*(plen+1))
            { Console.Printf("QWEN_ERROR invalid prompt length"); return false; }
            pids.Resize(plen);
            for (int i=0; i<plen; i++) pids[i]=LoadI32(blob,i+1);
        }
        if (plen<1 || plen+maxTokens-1>Q_MAXPOS)
        { Console.Printf("QWEN_ERROR prompt has %d tokens; limit is %d including template",plen,Q_MAXPOS-maxTokens+1); return false; }
        string line;
        for (int i=0; i<plen; i++)
        {
            if (pids[i]<0 || pids[i]>=Q_VOCAB)
            { Console.Printf("QWEN_ERROR invalid token ID"); return false; }
            line=line..String.Format(" %d",pids[i]);
        }
        Console.Printf("PIDS%s",line);
        return true;
    }

    bool LoadAll(out double ms)
    {
        double t0 = MSTimeF();
        if (weightsReady) { ms=0; return true; }
        loadFailed=false;
        int dl = Wads.CheckNumForFullName("data/decode.bin");
        if (dl<0) { Console.Printf("QWEN_ERROR missing decode.bin"); ms=0; return false; }
        string db = Wads.ReadLump(dl);
        decBlob = db;
        decOff.Resize(Q_VOCAB);
        int pos = 0, id = 0;
        int dlen = db.Length();
        while (pos < dlen && id < Q_VOCAB)
        {
            decOff[id] = pos;
            pos += 1 + db.ByteAt(pos);
            if (pos>dlen) { Console.Printf("QWEN_ERROR truncated decode table"); ms=0; return false; }
            id++;
        }

        if (id != Q_VOCAB || pos != dlen)
        { Console.Printf("QWEN_ERROR wrong decode vocabulary size"); ms=0; return false; }

        LoadW8("data/wq.bin", wq, Q_L * Q_D * Q_D);
        LoadW8("data/wk.bin", wk, Q_L * Q_D * Q_KVH * Q_HD);
        LoadW8("data/wv.bin", wv, Q_L * Q_D * Q_KVH * Q_HD);
        LoadW8("data/wo.bin", wo, Q_L * Q_D * Q_D);
        LoadW8("data/wg.bin", wg, Q_L * Q_D * Q_INTER);
        LoadW8("data/wu.bin", wu, Q_L * Q_D * Q_INTER);
        LoadW8("data/wd.bin", wd, Q_L * Q_INTER * Q_D);
        LoadW8("data/wte.bin", wte, Q_VOCAB * Q_D);

        LoadF64("data/sq.bin", sq, Q_L * Q_D);
        LoadF64("data/sk.bin", sk, Q_L * Q_KVH * Q_HD);
        LoadF64("data/sv.bin", sv, Q_L * Q_KVH * Q_HD);
        LoadF64("data/so.bin", so, Q_L * Q_D);
        LoadF64("data/sg.bin", sg, Q_L * Q_INTER);
        LoadF64("data/su.bin", su, Q_L * Q_INTER);
        LoadF64("data/sd.bin", sd, Q_L * Q_D);
        LoadF64("data/swte.bin", ste, Q_VOCAB);
        LoadF64("data/norm1.bin", n1, Q_L * Q_D);
        LoadF64("data/norm2.bin", n2, Q_L * Q_D);
        LoadF64("data/normf.bin", nf, Q_D);
        LoadF64("data/rope.bin", rope, 2 * Q_MAXPOS * 32);
        LoadF64("data/bq.bin", bq, Q_L * Q_D);
        LoadF64("data/bk.bin", bk, Q_L * Q_KVH * Q_HD);
        LoadF64("data/bv.bin", bv, Q_L * Q_KVH * Q_HD);

        bo.Resize(Q_L * Q_D);
        for (int i = 0; i < Q_L * Q_D; i++) bo[i] = 0.0;
        bg.Resize(Q_L * Q_INTER);
        for (int i = 0; i < Q_L * Q_INTER; i++) bg[i] = 0.0;
        bu.Resize(Q_L * Q_INTER);
        for (int i = 0; i < Q_L * Q_INTER; i++) bu[i] = 0.0;
        bd.Resize(Q_L * Q_D);
        for (int i = 0; i < Q_L * Q_D; i++) bd[i] = 0.0;
        bte.Resize(Q_VOCAB);
        for (int i = 0; i < Q_VOCAB; i++) bte[i] = 0.0;

        kc.Resize(Q_L * Q_KVH * Q_MAXPOS * Q_HD);
        vc.Resize(Q_L * Q_KVH * Q_MAXPOS * Q_HD);
        skc.Resize(Q_L * Q_KVH * Q_MAXPOS);
        svc.Resize(Q_L * Q_KVH * Q_MAXPOS);
        for (int i = 0; i < Q_L * Q_KVH * Q_MAXPOS; i++) { skc[i] = 1.0; svc[i] = 1.0; }

        x.Resize(Q_D); h.Resize(Q_D); qArr.Resize(Q_D);
        kArr.Resize(Q_KVH * Q_HD); vArr.Resize(Q_KVH * Q_HD);
        sil.Resize(Q_INTER); att.Resize(Q_D); hf.Resize(Q_D);
        upv.Resize(Q_INTER);
        logits.Resize(Q_VOCAB);
        xq.Resize(Q_D); ao.Resize(Q_D);
        xq2.Resize(Q_D); mq.Resize(Q_INTER); xqf.Resize(Q_D);
        e.Resize(Q_MAXPOS); p.Resize(Q_MAXPOS);
        kq8.Resize(Q_KVH * Q_HD); vq8.Resize(Q_KVH * Q_HD);
        ksc.Resize(Q_KVH); vsc.Resize(Q_KVH);
        lq.Resize(Q_L); lk.Resize(Q_L); lv.Resize(Q_L); lo.Resize(Q_L);
        lg.Resize(Q_L); lu.Resize(Q_L); ld.Resize(Q_L);

        ms = MSTimeF() - t0;
        weightsReady = !loadFailed;
        return weightsReady;
    }

    void LoadW8(string name, out Array<uint8> dst, int count)
    {
        int lump = Wads.CheckNumForFullName(name);
        if (lump<0 || Wads.GetLumpLength(lump)!=count)
        { Console.Printf("QWEN_ERROR missing or wrong size: %s",name); loadFailed=true; return; }
        string b = Wads.ReadLump(lump);
        dst.Resize(count);
        for (int i = 0; i < count; i++) dst[i] = b.ByteAt(i);
    }

    void LoadF64(string name, out Array<double> dst, int count)
    {
        int lump = Wads.CheckNumForFullName(name);
        if (lump<0 || Wads.GetLumpLength(lump)!=count*8)
        { Console.Printf("QWEN_ERROR missing or wrong size: %s",name); loadFailed=true; return; }
        string b = Wads.ReadLump(lump);
        dst.Resize(count);
        for (int i = 0; i < count; i++) dst[i] = LoadFx(b, i);
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

    double QuantArr(Array<double> src, int soff, Array<int> dst, int doff, int n)
    {
        double m = 0;
        for (int i = 0; i < n; i++)
        {
            double a = src[soff + i];
            if (a < 0) a = -a;
            if (a > m) m = a;
        }
        double s = m / 127.0;
        if (s <= 0) s = 1.0;
        for (int i = 0; i < n; i++)
            dst[doff + i] = RintEven(src[soff + i] / s);
        return s;
    }

    double ExpD(double x)
    {
        if (x < -700.0) return 0.0;
        double n = floor(x + 0.5);
        double f = x - n;
        double p = 1.5619206968586225e-16;
        p = p * f + 2.8114572543455206e-15;
        p = p * f + 4.779477332387385e-14;
        p = p * f + 7.647163731819816e-13;
        p = p * f + 1.1470745597729725e-11;
        p = p * f + 1.6059043836821613e-10;
        p = p * f + 2.08767569878681e-09;
        p = p * f + 2.505210838544172e-08;
        p = p * f + 2.755731922398589e-07;
        p = p * f + 2.7557319223985893e-06;
        p = p * f + 2.48015873015873e-05;
        p = p * f + 0.0001984126984126984;
        p = p * f + 0.001388888888888889;
        p = p * f + 0.008333333333333333;
        p = p * f + 0.041666666666666664;
        p = p * f + 0.16666666666666666;
        p = p * f + 0.5;
        p = p * f + 1.0;
        p = p * f + 1.0;
        int nn = n;
        double r = 1.0;
        if (nn >= 0)
            for (int i = 0; i < nn; i++) r *= 2.718281828459045;
        else
        {
            for (int i = 0; i < -nn; i++) r *= 2.718281828459045;
            r = 1.0 / r;
        }
        return r * p;
    }

    void MatvecQ(Array<uint8> w, int woff, Array<double> sc, int soff,
                 Array<double> bi, int boff, double xs,
                 Array<int> qin, int nin, int nout,
                 Array<double> y, out uint isum)
    {
        int sumx = 0;
        for (int j = 0; j < nin; j++) sumx += qin[j];
        isum = 0;
        for (int o = 0; o < nout; o++)
        {
            int accb = 0;
            int base = woff + o * nin;
            for (int j = 0; j < nin; j++)
            {
                int wb = w[base + j];
                accb += wb * qin[j];
            }
            int acc = accb - 128 * sumx;
            isum += acc;
            y[o] = sc[soff + o] * acc * xs + bi[boff + o];
        }
    }

    void RmsNorm(Array<double> src, int soff, Array<double> gam, int goff, Array<double> dst)
    {
        double ms = 0;
        for (int i = 0; i < Q_D; i++)
        {
            double v = src[soff + i];
            ms += v * v;
        }
        ms /= Q_D;
        double inv = 1.0 / sqrt(ms + Q_EPS);
        for (int i = 0; i < Q_D; i++)
            dst[i] = src[soff + i] * inv * gam[goff + i];
    }

    void ApplyRope(Array<double> arr, int off, int pos)
    {
        int cbase = pos * 32;
        int sbase = Q_MAXPOS * 32 + pos * 32;
        for (int j = 0; j < 32; j++)
        {
            double a = arr[off + j];
            double b = arr[off + 32 + j];
            double c = rope[cbase + j];
            double s = rope[sbase + j];
            arr[off + j] = a * c - b * s;
            arr[off + 32 + j] = a * s + b * c;
        }
    }

    string Detok(int tid)
    {
        if (tid<0 || tid>=decOff.Size()) return "";
        int off=decOff[tid];
        return decBlob.Mid(off+1,decBlob.ByteAt(off));
    }

    string EscapeBytes(string text)
    {
        string result;
        for (int i=0; i<int(text.Length()); i++)
        {
            int c=text.ByteAt(i);
            if (c>=32 && c<127 && c!=92) result=result..text.Mid(i,1);
            else result=result..String.Format("\\x%02X",c);
        }
        return result;
    }

    void PrintSums(string key, Array<uint> arr)
    {
        uint tot;
        for (int l = 0; l < Q_L; l++)
        {
            tot += arr[l];
            if (l == 0 || l == 12 || l == 23)
                Console.Printf("%s %d %.0f", key, l, double(arr[l]));
        }
        Console.Printf("%s 99 %.0f", key, double(tot));
    }

    int ForwardPos(int pos, int tid, bool needLogits)
    {
        for (int j = 0; j < Q_D; j++)
        {
            int wb = wte[tid * Q_D + j];
            x[j] = ste[tid] * (double(wb) - 128.0);
        }

        for (int l = 0; l < Q_L; l++)
        {
            RmsNorm(x, 0, n1, l * Q_D, h);
            double sx1 = QuantArr(h, 0, xq, 0, Q_D);
            uint part;
            MatvecQ(wq, l * Q_D * Q_D, sq, l * Q_D, bq, l * Q_D, sx1,
                    xq, Q_D, Q_D, qArr, part);
            lq[l] += part;
            MatvecQ(wk, l * Q_D * Q_KVH * Q_HD, sk, l * Q_KVH * Q_HD, bk, l * Q_KVH * Q_HD, sx1,
                    xq, Q_D, Q_KVH * Q_HD, kArr, part);
            lk[l] += part;
            MatvecQ(wv, l * Q_D * Q_KVH * Q_HD, sv, l * Q_KVH * Q_HD, bv, l * Q_KVH * Q_HD, sx1,
                    xq, Q_D, Q_KVH * Q_HD, vArr, part);
            lv[l] += part;

            for (int hh = 0; hh < Q_HEADS; hh++)
                ApplyRope(qArr, hh * Q_HD, pos);
            for (int hh = 0; hh < Q_KVH; hh++)
                ApplyRope(kArr, hh * Q_HD, pos);

            for (int hh = 0; hh < Q_KVH; hh++)
            {
                ksc[hh] = QuantArr(kArr, hh * Q_HD, kq8, hh * Q_HD, Q_HD);
                vsc[hh] = QuantArr(vArr, hh * Q_HD, vq8, hh * Q_HD, Q_HD);
            }
            int cb = (l * Q_KVH) * Q_MAXPOS * Q_HD;
            int cs = (l * Q_KVH) * Q_MAXPOS;
            for (int hh = 0; hh < Q_KVH; hh++)
            {
                for (int j = 0; j < Q_HD; j++)
                {
                    int a = kq8[hh * Q_HD + j];
                    int b = vq8[hh * Q_HD + j];
                    kc[cb + hh * Q_MAXPOS * Q_HD + pos * Q_HD + j] = a;
                    vc[cb + hh * Q_MAXPOS * Q_HD + pos * Q_HD + j] = b;
                }
                skc[cs + hh * Q_MAXPOS + pos] = ksc[hh];
                svc[cs + hh * Q_MAXPOS + pos] = vsc[hh];
            }
            for (int hh = 0; hh < Q_HEADS; hh++)
            {
                int kvh = hh / Q_GQA;
                int kb = cb + kvh * Q_MAXPOS * Q_HD;
                double m = 0;
                for (int u = 0; u <= pos; u++)
                {

                    double ks = skc[cs + kvh * Q_MAXPOS + u];
                    double qd = 0;
                    for (int j = 0; j < Q_HD; j++)
                    {
                        double kf = ks * kc[kb + u * Q_HD + j];
                        qd += kf * qArr[hh * Q_HD + j];
                    }
                    double sv = qd * 0.125;
                    e[u] = sv;
                    if (u == 0 || sv > m) m = sv;
                }
                double esum = 0;
                for (int u = 0; u <= pos; u++)
                {
                    e[u] = ExpD(e[u] - m);
                    esum += e[u];
                }
                for (int u = 0; u <= pos; u++)
                    p[u] = e[u] / esum;
                for (int j = 0; j < Q_HD; j++)
                {
                    double acc = 0;
                    for (int u = 0; u <= pos; u++)
                        acc += p[u] * (svc[cs + kvh * Q_MAXPOS + u] *
                               vc[cb + kvh * Q_MAXPOS * Q_HD + u * Q_HD + j]);
                    att[hh * Q_HD + j] = acc;
                }
            }

            double sxo = QuantArr(att, 0, ao, 0, Q_D);
            MatvecQ(wo, l * Q_D * Q_D, so, l * Q_D, bo, l * Q_D, sxo,
                    ao, Q_D, Q_D, h, part);
            lo[l] += part;
            for (int j = 0; j < Q_D; j++) x[j] += h[j];

            RmsNorm(x, 0, n2, l * Q_D, h);
            double sx2 = QuantArr(h, 0, xq2, 0, Q_D);
            MatvecQ(wg, l * Q_D * Q_INTER, sg, l * Q_INTER, bg, l * Q_INTER, sx2,
                    xq2, Q_D, Q_INTER, sil, part);
            lg[l] += part;
            MatvecQ(wu, l * Q_D * Q_INTER, su, l * Q_INTER, bu, l * Q_INTER, sx2,
                    xq2, Q_D, Q_INTER, upv, part);
            lu[l] += part;
            for (int i = 0; i < Q_INTER; i++)
            {
                double g = sil[i];
                sil[i] = g / (1.0 + ExpD(-g)) * upv[i];
            }
            double sxm = QuantArr(sil, 0, mq, 0, Q_INTER);
            MatvecQ(wd, l * Q_INTER * Q_D, sd, l * Q_D, bd, l * Q_D, sxm,
                    mq, Q_INTER, Q_D, h, part);
            ld[l] += part;
            for (int j = 0; j < Q_D; j++) x[j] += h[j];
        }

        if (!needLogits) return -1;

        RmsNorm(x, 0, nf, 0, hf);
        double sxf = QuantArr(hf, 0, xqf, 0, Q_D);
        uint acc;
        MatvecQ(wte, 0, ste, 0, bte, 0, sxf, xqf, Q_D, Q_VOCAB, logits, acc);
        if (!logit0Done)
        {
            logit0Sum = acc;
        }
        int best = 0;
        double bv = logits[0];
        for (int v = 1; v < Q_VOCAB; v++)
            if (logits[v] > bv) { bv = logits[v]; best = v; }
        if (!logit0Done)
        {
            logit0Done = true;
            logit0Id = best;
            logit0Max = bv;
        }
        return best;
    }
}
