

class LFMTokenizer : Object
{
    Array<int> byteIds, leftIds, rightIds, resultIds, ranks;
    Array<int> rangeLo, rangeHi, rangeKind, specialIds;
    Array<string> specials;
    bool ready;
    Array<int> wordSlots, decOff;
    string decBlob;

    int ReadInt(string b, int i)
    {
        return b.ByteAt(i) | (b.ByteAt(i+1)<<8) |
            (b.ByteAt(i+2)<<16) | (b.ByteAt(i+3)<<24);
    }

    int HashPair(int a, int b)
    {
        return int((uint(a) * 1664525u + uint(b) * 1013904223u) & 524287u);
    }

    bool Load()
    {
        if (ready) return true;
        int lump = Wads.CheckNumForFullName("data/bpe.bin");
        int ul = Wads.CheckNumForFullName("data/unicode.bin");
        int sl = Wads.CheckNumForFullName("data/special.txt");
        int si = Wads.CheckNumForFullName("data/special.bin");
        if (lump < 0 || ul < 0 || sl < 0 || si < 0)
        { Console.Printf("LLM_ERROR missing tokenizer tables; run prep_prompt.py"); return false; }
        string b = Wads.ReadLump(lump);
        if (b.Length() < 1028) return false;
        int count = ReadInt(b, 1024);
        if (count < 0 || count > 400000 || b.Length() != 1028 + count*12) return false;
        byteIds.Resize(256);
        for (int i=0; i<256; i++) byteIds[i] = ReadInt(b, i*4);
        leftIds.Resize(524288); rightIds.Resize(524288);
        resultIds.Resize(524288); ranks.Resize(524288);
        for (int i=0; i<524288; i++) ranks[i] = -1;
        for (int i=0; i<count; i++)
        {
            int a = ReadInt(b, 1028+i*12), c = ReadInt(b, 1032+i*12);
            int h = HashPair(a,c);
            while (ranks[h] >= 0) h = (h+1)&524287;
            leftIds[h]=a; rightIds[h]=c; resultIds[h]=ReadInt(b,1036+i*12); ranks[h]=i;
        }
        b = Wads.ReadLump(ul);
        if (b.Length()%12 != 0) return false;
        count = b.Length()/12;
        rangeLo.Resize(count); rangeHi.Resize(count); rangeKind.Resize(count);
        for (int i=0; i<count; i++)
        { rangeLo[i]=ReadInt(b,i*12); rangeHi[i]=ReadInt(b,i*12+4); rangeKind[i]=ReadInt(b,i*12+8); }
        b = Wads.ReadLump(sl);
        b.Split(specials,"\n");
        b = Wads.ReadLump(si);
        if (b.Length() != specials.Size()*4) return false;
        specialIds.Resize(specials.Size());
        for (int i=0; i<specials.Size(); i++) specialIds[i]=ReadInt(b,i*4);
        int dl=Wads.CheckNumForFullName("data/decode.bin");
        if(dl<0) return false;
        decBlob=Wads.ReadLump(dl);
        decOff.Resize(128000);
        wordSlots.Resize(524288);
        for(int i=0;i<524288;i++) wordSlots[i]=-1;
        int off=0;
        for(int i=0;i<128000;i++)
        {
            if(off>=int(decBlob.Length())) return false;
            decOff[i]=off;
            int len=decBlob.ByteAt(off);
            if(off+1+len>int(decBlob.Length())) return false;
            if(len>0)
            {
                string word=decBlob.Mid(off+1,len);
                int h=HashWord(word);
                while(wordSlots[h]>=0) h=(h+1)&524287;
                wordSlots[h]=i;
            }
            off+=1+len;
        }
        if(off!=decBlob.Length()) return false;
        ready=true;
        return true;
    }

    int HashWord(string word)
    {
        uint h=2166136261u;
        for(int i=0;i<int(word.Length());i++) h=(h^uint(word.ByteAt(i)))*16777619u;
        return int(h&524287u);
    }

    string Decode(int tid)
    {
        if(tid<0 || tid>=decOff.Size()) return "";
        int off=decOff[tid];
        return decBlob.Mid(off+1,decBlob.ByteAt(off));
    }

    int WordId(string word)
    {
        int h=HashWord(word);
        while(wordSlots[h]>=0)
        {
            int tid=wordSlots[h];
            if(Decode(tid)==word) return tid;
            h=(h+1)&524287;
        }
        return -1;
    }

    int Kind(int cp)
    {
        int lo=0, hi=rangeLo.Size()-1;
        while (lo<=hi)
        {
            int mid=(lo+hi)/2;
            if (cp<rangeLo[mid]) hi=mid-1;
            else if (cp>rangeHi[mid]) lo=mid+1;
            else return rangeKind[mid];
        }
        return 0;
    }

    int PairSlot(int a, int b)
    {
        int h=HashPair(a,b);
        while (ranks[h]>=0)
        {
            if (leftIds[h]==a && rightIds[h]==b) return h;
            h=(h+1)&524287;
        }
        return -1;
    }

    void Piece(string text, Array<int> ids)
    {

        int whole=WordId(text);
        if(whole>=0) { ids.Push(whole); return; }
        Array<int> work;
        work.Resize(text.Length());
        for (int i=0; i<int(text.Length()); i++) work[i]=byteIds[text.ByteAt(i)];
        while (work.Size()>1)
        {
            int best=-1, bestRank=2147483647, merged=-1;
            for (int i=0; i<work.Size()-1; i++)
            {
                int h=PairSlot(work[i],work[i+1]);
                if (h>=0 && ranks[h]<bestRank)
                { best=i; bestRank=ranks[h]; merged=resultIds[h]; }
            }
            if (best<0) break;
            work[best]=merged;
            for (int i=best+1; i<work.Size()-1; i++) work[i]=work[i+1];
            work.Resize(work.Size()-1);
        }
        for (int i=0; i<work.Size(); i++) ids.Push(work[i]);
    }

    void Ordinary(string text, Array<int> ids)
    {
        Array<int> cp, offsets, kinds;
        int offset=0;
        while (offset<int(text.Length()))
        {
            offsets.Push(offset);
            int c;
            [c, offset]=text.GetNextCodePoint(offset);
            cp.Push(c); kinds.Push(Kind(c));
        }
        offsets.Push(text.Length());
        int n=cp.Size(), i=0;
        while (i<n)
        {
            int end=i;
            if (cp[i]==39)
            {
                string tail=text.Mid(offsets[i],3).MakeLower();
                if (tail.Left(2)=="'s" || tail.Left(2)=="'t" || tail.Left(2)=="'m" || tail.Left(2)=="'d") end=i+2;
                else if (tail=="'re" || tail=="'ve" || tail=="'ll") end=i+3;
            }
            if (end==i)
            {
                int j=i;
                if (kinds[i]!=1 && kinds[i]!=2 && cp[i]!=10 && cp[i]!=13) j++;
                if (j<n && kinds[j]==1)
                { while (j<n && kinds[j]==1) j++; end=j; }
            }
            if (end==i && kinds[i]==2)
            { end=i+1; while(end<n && end<i+3 && kinds[end]==2) end++; }
            if (end==i)
            {
                int j=i;
                if (cp[j]==32) j++;
                int start=j;
                while (j<n && kinds[j]==0) j++;
                if (j>start)
                { while (j<n && (cp[j]==10 || cp[j]==13)) j++; end=j; }
            }
            if (end==i && kinds[i]==3)
            {
                int j=i, lastNewline=-1;
                while (j<n && kinds[j]==3)
                { if (cp[j]==10 || cp[j]==13) lastNewline=j; j++; }
                if (lastNewline>=0) end=lastNewline+1;
                else if (j==n) end=j;
                else end=max(i+1,j-1);
            }
            if (end==i) end=i+1;
            Piece(text.Mid(offsets[i],offsets[end]-offsets[i]),ids);
            i=end;
        }
    }

    bool Encode(string text, Array<int> ids)
    {
        ids.Clear();
        if (!Load()) { Console.Printf("LLM_ERROR tokenizer table validation failed"); return false; }
        int pos=0;
        while (pos<int(text.Length()))
        {
            int at=text.Length(), token=-1;
            for (int i=0; i<specials.Size(); i++)
            {
                int found=text.IndexOf(specials[i],pos);
                if (found>=0 && found<at) { at=found; token=i; }
            }
            Ordinary(text.Mid(pos,at-pos),ids);
            if (token<0) break;
            ids.Push(specialIds[token]);
            pos=at+specials[token].Length();
        }
        return true;
    }
}
