
class LFMMatrix : Object
{
    Array<uint8> bytes;
    Array<double> scale;
    int rows, cols, loaded;
    string path, blob;
    bool failed;

    bool Prepare(string stem, int nr, int nc)
    {
        path="data/"..stem..".bin"; rows=nr; cols=nc;
        int lump=Wads.CheckNumForFullName(path);
        int sl=Wads.CheckNumForFullName("data/"..stem..".scale");
        if(lump<0 || sl<0 || Wads.GetLumpLength(lump)!=rows*cols || Wads.GetLumpLength(sl)!=rows*8)
        { Console.Printf("LLM_ERROR missing or wrong tensor size: %s",path); failed=true; return false; }
        string sb=Wads.ReadLump(sl);
        scale.Resize(rows);
        for(int i=0;i<rows;i++) scale[i]=LFMRuntime.F64(sb,i);
        return true;
    }

    bool LoadStep()
    {
        if(loaded==rows*cols) return true;
        if(loaded==0)
        {
            blob=Wads.ReadLump(Wads.CheckNumForFullName(path));
            bytes.Resize(rows*cols);
        }
        int end=min(loaded+131072,rows*cols);
        for(int i=loaded;i<end;i++) bytes[i]=blob.ByteAt(i);
        loaded=end;
        if(loaded==rows*cols) { blob=""; return true; }
        return false;
    }
}

class LFMKVPage : Object
{
    Array<double> keys, values;
}

class LFMBlock : Object
{
    int kind;
    LFMMatrix ip, op, q, k, v, gate, w1, w2, w3;
    Array<double> norm, postNorm, ffnNorm, postFfnNorm, qNorm, kNorm, kernel;
    Array<double> history;
    Array<LFMKVPage> pages;
}

class LFMRuntime : Object
{
    Array<LFMMatrix> matrices;
    Array<LFMBlock> layers;
    LFMMatrix embedding;
    Array<double> finalNorm, ropeFull, ropeSliding;
    Array<double> x,h,proj,mix,result,tmp,qarr,karr,varr,gates,silu,up,logits,scores;
    int loadIndex, pos, layer, stage, row, head, nextId;
    int attPhase, attCursor, attFirst;
    double attPeak, attTotal;
    double nextValue, mvBias;
    bool error, ready, needLogits, debug;
    bool profile;
    Array<double> stageMs;
    Array<int> stageCalls;

    static double F64(string b,int index)
    {
        int i=index*8;
        uint lo=b.ByteAt(i)|(b.ByteAt(i+1)<<8)|(b.ByteAt(i+2)<<16)|(b.ByteAt(i+3)<<24);
        uint hi=b.ByteAt(i+4)|(b.ByteAt(i+5)<<8)|(b.ByteAt(i+6)<<16)|(b.ByteAt(i+7)<<24);
        int exponent=int((hi>>20)&2047u);
        double mantissa=double(hi&1048575u)*4294967296.0+double(lo);
        double value=exponent==0 ? mantissa : 1.0+mantissa/4503599627370496.0;
        int power=exponent==0 ? -1074 : exponent-1023;
        if(power>=0) for(int j=0;j<power;j++) value*=2.0;
        else for(int j=0;j<-power;j++) value*=0.5;
        return (hi&2147483648u)!=0 ? -value : value;
    }

    void Vector(string stem,Array<double> target,int count)
    {
        int lump=Wads.CheckNumForFullName("data/"..stem..".bin");
        if(lump<0 || Wads.GetLumpLength(lump)!=count*8)
        { Console.Printf("LLM_ERROR missing or wrong vector: %s",stem); error=true; return; }
        string b=Wads.ReadLump(lump);
        target.Resize(count);
        for(int i=0;i<count;i++) target[i]=F64(b,i);
    }

    LFMMatrix Matrix(string stem,int nr,int nc)
    {
        let m=new("LFMMatrix");
        if(!m.Prepare(stem,nr,nc)) error=true;
        matrices.Push(m);
        return m;
    }

    virtual bool Configure() { return false; }

    bool LoadStep()
    {
        if(error) return false;
        if(loadIndex<matrices.Size())
        {
            if(matrices[loadIndex].LoadStep()) loadIndex++;
            return false;
        }
        if(!ready)
        {
            x.Resize(2048); h.Resize(2048); proj.Resize(6144); mix.Resize(2048);
            result.Resize(2048); tmp.Resize(2048); qarr.Resize(2048);
            karr.Resize(512); varr.Resize(512); gates.Resize(16);
            silu.Resize(7680); up.Resize(7680); logits.Resize(128000); scores.Resize(512);
            for(int l=0;l<24;l++)
            {
                if(layers[l].kind==0) layers[l].history.Resize(7*2048);

            }
            ready=true;
        }
        return true;
    }

    void Reset()
    {
        if(profile)
        {
            stageMs.Resize(50); stageCalls.Resize(50);
            for(int i=0;i<50;i++) { stageMs[i]=0; stageCalls[i]=0; }
        }
        for(int l=0;l<24;l++)
            if(layers[l].kind==0)
                for(int i=0;i<layers[l].history.Size();i++) layers[l].history[i]=0;
        stage=99; row=0; pos=0;

        for(int l=0;l<24;l++) layers[l].pages.Clear();
    }

    void StartToken(int tid,int position,bool wantLogits)
    {
        pos=position; needLogits=wantLogits; layer=0; row=0; head=0; stage=10;
        double s=embedding.scale[tid];
        for(int j=0;j<2048;j++) x[j]=(int(embedding.bytes[tid*2048+j])-128)*s;
    }

    void Norm(Array<double> src,int offset,Array<double> scale,Array<double> dst,int dest,int count)
    {
        double sum=0;
        for(int j=0;j<count;j++) { double v=src[offset+j]; sum+=v*v; }
        double inv=1.0/sqrt(sum/count+0.00001);
        for(int j=0;j<count;j++) dst[dest+j]=src[offset+j]*inv*scale[j];
    }

    void DotRows(Array<uint8> weights, Array<double> scales, Array<double> input,
                 Array<double> output, int cols, int first, int end)
    {
        for(int o=first;o<end;o++)
        {
            double a=0,b=0,c=0,d=0;
            int base=o*cols;
            for(int j=0;j<cols;j+=4)
            {
                a+=weights[base+j]*input[j];
                b+=weights[base+j+1]*input[j+1];
                c+=weights[base+j+2]*input[j+2];
                d+=weights[base+j+3]*input[j+3];
            }
            output[o]=((a+b)+(c+d)-mvBias)*scales[o];
        }
    }

    bool MV(LFMMatrix m,Array<double> input,Array<double> output)
    {


        if(row==0)
        {
            double sum=0;
            for(int j=0;j<m.cols;j++) sum+=input[j];
            mvBias=128.0*sum;
        }
        int end=min(row+16,m.rows);
        DotRows(m.bytes,m.scale,input,output,m.cols,row,end);
        row=end;
        if(row==m.rows) { row=0; return true; }
        return false;
    }

    void Rope(Array<double> values,int count,Array<double> table,bool full=false)
    {


        for(int j=0;j<64;j++)
        {
            double c,s;
            if(pos<512) { c=table[pos*64+j]; s=table[512*64+pos*64+j]; }
            else
            {
                double inv=full && j>=32 ? 0 : exp(-double(j)/64.0*(full ? 13.815510557964274 : 9.210340371976184));


                double angle=double(pos)*inv;
                angle-=floor(angle/6.283185307179586+0.5)*6.283185307179586;
                double square=angle*angle, ct=1, st=angle;
                c=ct; s=st;
                for(int n=1;n<=13;n++)
                {
                    ct*=-square/((2*n-1)*(2*n));
                    st*=-square/((2*n)*(2*n+1));
                    c+=ct; s+=st;
                }
            }
            for(int hh=0;hh<count;hh++)
            {
                double a=values[hh*128+j], b=values[hh*128+64+j];
                values[hh*128+j]=a*c-b*s;
                values[hh*128+64+j]=a*s+b*c;
            }
        }
    }

    void Cache(LFMBlock b)
    {
        int page=b.kind==2 ? 0 : pos/256;
        if(page>=b.pages.Size())
        {
            let p=new("LFMKVPage");
            p.keys.Resize(256*512); p.values.Resize(256*512);
            b.pages.Push(p);
        }
        let p=b.pages[page];
        int offset=(pos%256)*512;
        for(int j=0;j<512;j++) { p.keys[offset+j]=karr[j]; p.values[offset+j]=varr[j]; }
    }

    bool AttentionStep(LFMBlock b,int hh)
    {
        int kv=hh/4;
        if(attPhase==0)
        {
            attFirst=b.kind==2 ? max(0,pos-255) : 0;
            if(scores.Size()<pos-attFirst+1) scores.Resize(((pos-attFirst)/256+1)*256);
            attPeak=-1e300; attTotal=0; attCursor=attFirst; attPhase=1;
            for(int j=0;j<128;j++) mix[hh*128+j]=0;
        }
        if(attPhase==1)
        {
            int end=min(pos+1,attCursor+16);
            for(int u=attCursor;u<end;u++)
            {
                let p=b.pages[b.kind==2 ? 0 : u/256];
                int offset=(u%256)*512+kv*128;
                double acc=0;
                for(int j=0;j<128;j++) acc+=qarr[hh*128+j]*p.keys[offset+j];
                scores[u-attFirst]=acc*0.08838834764831845;
                attPeak=max(attPeak,scores[u-attFirst]);
            }
            attCursor=end;
            if(attCursor>pos) { attPhase=2; attCursor=attFirst; }
            return false;
        }
        if(attPhase==2)
        {
            int end=min(pos+1,attCursor+256);
            for(int u=attCursor;u<end;u++)
            {
                scores[u-attFirst]=exp(scores[u-attFirst]-attPeak);
                attTotal+=scores[u-attFirst];
            }
            attCursor=end;
            if(attCursor>pos) { attPhase=3; attCursor=attFirst; }
            return false;
        }
        int end=min(pos+1,attCursor+16);
        for(int u=attCursor;u<end;u++)
        {
            let p=b.pages[b.kind==2 ? 0 : u/256];
            int offset=(u%256)*512+kv*128;
            double weight=scores[u-attFirst]/attTotal;
            for(int j=0;j<128;j++) mix[hh*128+j]+=weight*p.values[offset+j];
        }
        attCursor=end;
        if(attCursor<=pos) return false;
        double gate=1.0/(1.0+exp(-gates[hh]));
        for(int j=0;j<128;j++) mix[hh*128+j]*=gate;
        attPhase=0;
        return true;
    }

    void Trace()
    {
        if(!debug || !needLogits || (layer!=0 && layer!=1 && layer!=2 && layer!=4 && layer!=23)) return;
        for(int start=0;start<2048;start+=16)
        {
            string line=String.Format("VEC %d %d %d",pos,layer,start);
            for(int j=0;j<16;j++) line=line..String.Format(" %.17g",x[start+j]);
            Console.Printf("%s",line);
        }
    }

    bool Step()
    {
        if(stage==99) return true;
        int measuredStage=stage;
        double measuredStart=profile ? MSTimeF() : 0;
        LFMBlock b=layers[min(layer,23)];
        switch(stage)
        {
        case 10:
            Norm(x,0,b.norm,h,0,2048);
            stage=b.kind==0 ? 11 : 20;
            break;
        case 11: if(MV(b.ip,h,proj)) stage=12; break;
        case 12:
            for(int j=0;j<2048;j++) b.history[(pos%7)*2048+j]=proj[j]*proj[4096+j];
            for(int j=0;j<2048;j++)
            {
                double acc=0;
                for(int k=0;k<7;k++)
                {
                    int u=pos-6+k;
                    if(u>=0) acc+=b.history[(u%7)*2048+j]*b.kernel[j*7+k];
                }
                mix[j]=proj[2048+j]*acc;
            }
            stage=13; break;
        case 13: if(MV(b.op,mix,result)) stage=30; break;
        case 20: if(MV(b.q,h,qarr)) stage=21; break;
        case 21: if(MV(b.k,h,karr)) stage=22; break;
        case 22: if(MV(b.v,h,varr)) stage=23; break;
        case 23: if(MV(b.gate,h,gates)) stage=24; break;
        case 24:
            for(int hh=0;hh<16;hh++) Norm(qarr,hh*128,b.qNorm,qarr,hh*128,128);
            for(int hh=0;hh<4;hh++) Norm(karr,hh*128,b.kNorm,karr,hh*128,128);
            if(b.kind==1) { Rope(qarr,16,ropeFull,true); Rope(karr,4,ropeFull,true); }
            else { Rope(qarr,16,ropeSliding); Rope(karr,4,ropeSliding); }
            Cache(b);
            head=0; attPhase=0; stage=25; break;
        case 25: if(AttentionStep(b,head)) head++; if(head==16) stage=26; break;
        case 26: if(MV(b.op,mix,result)) stage=30; break;
        case 30:
            Norm(result,0,b.postNorm,tmp,0,2048);
            for(int j=0;j<2048;j++) x[j]+=tmp[j];
            Norm(x,0,b.ffnNorm,h,0,2048);
            stage=31; break;
        case 31: if(MV(b.w1,h,silu)) stage=32; break;
        case 32: if(MV(b.w3,h,up)) stage=33; break;
        case 33:
            for(int j=0;j<7680;j++) silu[j]=(silu[j]/(1.0+exp(-silu[j])))*up[j];
            stage=34; break;
        case 34: if(MV(b.w2,silu,result)) stage=35; break;
        case 35:
            Norm(result,0,b.postFfnNorm,tmp,0,2048);
            for(int j=0;j<2048;j++) x[j]+=tmp[j];
            Trace();
            layer++;
            if(layer<24) stage=10;
            else if(needLogits) { Norm(x,0,finalNorm,h,0,2048); stage=40; }
            else { nextId=-1; stage=99; }
            break;
        case 40: if(MV(embedding,h,logits)) stage=41; break;
        case 41:
            nextId=0; nextValue=logits[0];
            for(int i=1;i<128000;i++) if(logits[i]>nextValue) { nextId=i; nextValue=logits[i]; }
            stage=99; break;
        }
        if(profile)
        {
            stageMs[measuredStage]+=MSTimeF()-measuredStart;
            stageCalls[measuredStage]++;
        }
        return stage==99;
    }
}
