version "4.14"

class MatmulBench : StaticEventHandler
{
    Array<int> W;
    Array<uint8> W8;
    Array<int> xv;
    Array<int> yv;
    bool benchRan;
    int resumeTicks;

    override void WorldLoaded(WorldEvent e)
    {
        Console.Printf("WORLD_LOADED_OK");
        if (benchRan) return;
        int specLump = Wads.CheckNumForFullName("data/benchspec.txt");
        if (specLump < 0) return;

        Array<int> sizes;
        Array<int> modes;
        ParseSpec(Wads.ReadLump(specLump), sizes, modes);
        for (int i = 0; i < sizes.Size(); i++) RunSize(sizes[i], modes[i]);

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
        if (e.Name ~== "matmul" && e.Args[0] > 0)
            RunSize(e.Args[0], e.Args[1] >= 0 ? e.Args[1] : 0);
    }

    void ParseSpec(string s, out Array<int> sizes, out Array<int> modes)
    {
        int cur = -1;
        for (int i = 0; i < s.Length(); i++)
        {
            int c = s.ByteAt(i);
            if (c >= 48 && c <= 57)
            {
                if (cur < 0) cur = 0;
                cur = cur * 10 + (c - 48);
            }
            else if (cur > 0)
            {
                sizes.Push(cur / 10);
                modes.Push(cur % 10);
                cur = -1;
            }
        }
        if (cur > 0)
        {
            sizes.Push(cur / 10);
            modes.Push(cur % 10);
        }
    }

    bool LoadData(int n, int mode)
    {
        int wl = Wads.CheckNumForFullName(String.Format("data/w%d.bin", n));
        int xl = Wads.CheckNumForFullName(String.Format("data/x%d.bin", n));
        if (wl < 0 || xl < 0)
        {
            Console.Printf("MISSING_LUMPS size=%d", n);
            return false;
        }

        double t0 = MSTimeF();
        string wblob = Wads.ReadLump(wl);
        string xblob = Wads.ReadLump(xl);

        int n2 = n * n;
        if (mode == 0)
        {
            W.Resize(n2);
            for (int i = 0; i < n2; i++)
            {
                int b = wblob.ByteAt(i);
                W[i] = b - 128;
            }
        }
        else
        {
            W8.Resize(n2);
            for (int i = 0; i < n2; i++)
                W8[i] = wblob.ByteAt(i);
        }
        xv.Resize(n);
        for (int i = 0; i < n; i++)
            xv[i] = int(xblob.ByteAt(i)) - 128;
        yv.Resize(n);

        double t1 = MSTimeF();
        Console.Printf("LOAD size=%d mode=%d bytes=%d ms=%.3f", n, mode, n2 + n, t1 - t0);
        return true;
    }

    void MatvecI32(int n)
    {
        int wi = 0;
        for (int i = 0; i < n; i++)
        {
            int acc = 0;
            for (int j = 0; j < n; j++)
                acc += W[wi + j] * xv[j];
            yv[i] = acc;
            wi += n;
        }
    }

    void MatvecU8(int n)
    {
        int sumx = 0;
        for (int j = 0; j < n; j++) sumx += xv[j];

        int wi = 0;
        for (int i = 0; i < n; i++)
        {
            int acc = 0;
            for (int j = 0; j < n; j++)
            {
                int wb = W8[wi + j];
                acc += wb * xv[j];
            }
            yv[i] = acc - 128 * sumx;
            wi += n;
        }
    }

    void RunSize(int n, int mode)
    {
        if (mode == 2) { RunSize(n, 0); RunSize(n, 1); return; }
        if (!LoadData(n, mode)) return;

        double t0 = MSTimeF();
        if (mode == 0) MatvecI32(n); else MatvecU8(n);
        double est = MSTimeF() - t0;

        int iters = 3;
        if (est < 2.0) iters = 200;
        else if (est < 20.0) iters = 50;
        else if (est < 200.0) iters = 10;

        double t1 = MSTimeF();
        for (int i = 0; i < iters; i++)
            if (mode == 0) MatvecI32(n); else MatvecU8(n);
        double total = MSTimeF() - t1;
        double per = total / iters;
        double mps = (double(n) * double(n) * iters) / (total / 1000.0);

        uint sum = 0;
        for (int i = 0; i < n; i++) sum += yv[i];

        Console.Printf("RESULT size=%d mode=%d iters=%d warmup_ms=%.3f total_ms=%.3f ms_per_iter=%.4f macs_per_sec=%.1f",
            n, mode, iters, est, total, per, mps);
        Console.Printf("Y size=%d mode=%d sum=%.0f head=%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
            n, mode, double(sum),
            yv[0], yv[1], yv[2], yv[3], yv[4], yv[5], yv[6], yv[7],
            yv[8], yv[9], yv[10], yv[11], yv[12], yv[13], yv[14], yv[15]);
    }
}
