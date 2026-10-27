

class QwenTokenizer : Object
{
    Array<int> byteIds, leftIds, rightIds, resultIds, ranks;
    Array<int> rangeLo, rangeHi, rangeKind, specialIds;
    Array<string> specials;
    bool ready;
    Array<int> normCp, normClass, normOffset, normLength, normData;
    Array<int> compA, compB, compResult;

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
        { Console.Printf("QWEN_ERROR missing tokenizer tables; run prep_prompt.py"); return false; }
        string b = Wads.ReadLump(lump);
        if (b.Length() < 1028) return false;
        int count = ReadInt(b, 1024);
        if (count < 0 || count > 200000 || b.Length() != 1028 + count*12) return false;
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
        int nl=Wads.CheckNumForFullName("data/nfc.bin");
        if (nl<0) { Console.Printf("QWEN_ERROR missing NFC table"); return false; }
        b=Wads.ReadLump(nl);
        count=ReadInt(b,0);
        normCp.Resize(count); normClass.Resize(count); normOffset.Resize(count); normLength.Resize(count);
        normData.Clear();
        int off=4;
        for (int i=0;i<count;i++)
        {
            normCp[i]=ReadInt(b,off); normClass[i]=ReadInt(b,off+4);
            int len=ReadInt(b,off+8); off+=12;
            normLength[i]=len; normOffset[i]=normData.Size();
            for (int j=0;j<len;j++) { normData.Push(ReadInt(b,off)); off+=4; }
        }
        count=ReadInt(b,off); off+=4;
        compA.Resize(count); compB.Resize(count); compResult.Resize(count);
        for (int i=0;i<count;i++)
        { compA[i]=ReadInt(b,off); compB[i]=ReadInt(b,off+4); compResult[i]=ReadInt(b,off+8); off+=12; }
        if (off!=b.Length()) return false;
        ready=true;
        return true;
    }

    int NormIndex(int cp)
    {
        int lo=0, hi=normCp.Size()-1;
        while(lo<=hi)
        {
            int mid=(lo+hi)/2;
            if(cp<normCp[mid]) hi=mid-1;
            else if(cp>normCp[mid]) lo=mid+1;
            else return mid;
        }
        return -1;
    }

    int CombiningClass(int cp)
    {
        int idx=NormIndex(cp);
        return idx<0 ? 0 : normClass[idx];
    }

    int Compose(int a, int b)
    {

        if(a>=0x1100 && a<0x1113 && b>=0x1161 && b<0x1176)
            return 0xAC00+((a-0x1100)*21+b-0x1161)*28;
        if(a>=0xAC00 && a<0xD7A4 && (a-0xAC00)%28==0 && b>0x11A7 && b<0x11C3)
            return a+b-0x11A7;
        int lo=0,hi=compA.Size()-1;
        while(lo<=hi)
        {
            int mid=(lo+hi)/2;
            if(a<compA[mid] || (a==compA[mid] && b<compB[mid])) hi=mid-1;
            else if(a>compA[mid] || (a==compA[mid] && b>compB[mid])) lo=mid+1;
            else return compResult[mid];
        }
        return -1;
    }

    string Normalize(string text)
    {
        Array<int> decomposed, composed;
        int offset=0;
        while(offset<int(text.Length()))
        {
            int cp;
            [cp,offset]=text.GetNextCodePoint(offset);
            int idx=NormIndex(cp);
            if(idx<0) decomposed.Push(cp);
            else for(int j=0;j<normLength[idx];j++) decomposed.Push(normData[normOffset[idx]+j]);
        }

        for(int i=1;i<decomposed.Size();i++)
        {
            int cp=decomposed[i], cc=CombiningClass(cp), j=i;
            if(cc==0) continue;
            while(j>0 && CombiningClass(decomposed[j-1])>cc)
            { decomposed[j]=decomposed[j-1]; j--; }
            decomposed[j]=cp;
        }
        int starter=-1, lastClass=0;
        for(int i=0;i<decomposed.Size();i++)
        {
            int cp=decomposed[i], cc=CombiningClass(cp), combined=-1;
            if(starter>=0 && (lastClass<cc || lastClass==0)) combined=Compose(composed[starter],cp);
            if(combined>=0) composed[starter]=combined;
            else
            {
                if(cc==0) starter=composed.Size();
                composed.Push(cp); lastClass=cc;
            }
        }
        string result;
        for(int i=0;i<composed.Size();i++) result.AppendCharacter(composed[i]);
        return result;
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
            if (end==i && kinds[i]==2) end=i+1;
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
        if (text.Length()>8192)
        { Console.Printf("QWEN_ERROR prompt exceeds 8192 UTF-8 bytes"); return false; }
        if (!Load()) return false;
        text=Normalize(text);
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
