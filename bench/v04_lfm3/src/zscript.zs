version "4.14"
#include "tokenizer.zs"
#include "runtime.zs"
#include "model.zs"


class LFMRunner : StaticEventHandler
{
    ui LFMRuntime model;
    ui LFMTokenizer tokenizer;
    ui Array<int> pids;
    ui int mode, position, generated, maxTokens, renderCount, renderSlow;
    ui bool autoChecked, debug, renderWasRunning, thinking;
    ui string reply, pendingUtf8;
    ui double started, lastReport, loadStarted, maxSlice, maxInferenceSlice, computeMs, lastRender, renderTotal, renderMax, prefillStarted;
    ui double loadActiveMs, setupMs;

    override void ConsoleProcess(ConsoleEvent e)
    {
        if(e.Name ~== "llm") Start();
        else if(e.Name ~== "llm_cancel") Cancel();
        else if(e.Name ~== "llm_tokenize")
        {
            if(mode!=0) { Console.Printf("LFM3 is working; cancel it before tokenizing another prompt."); return; }
            bool wasDebug=debug; debug=true; Tokenize(); debug=wasDebug;
        }
    }

    ui void Cancel()
    {
        if(mode!=0)
        { mode=0; Console.Printf("LFM3: cancelled. %s",reply); Console.Printf("LLM_CANCELLED"); }
    }

    ui bool Tokenize()
    {
        if(tokenizer==null) tokenizer=new("LFMTokenizer");
        string query=CVar.GetCVar('llm_prompt').GetString();
        string system=CVar.GetCVar('llm_system').GetString();
        if(query.Length()==0) { Console.Printf("Set llm_prompt \"your question\", then enter llm."); return false; }
        string text="<|startoftext|>";
        if(system.Length()>0) text=text.."<|im_start|>system\n"..system.."<|im_end|>\n";
        thinking=CVar.GetCVar('llm_thinking').GetBool();
        text=text.."<|im_start|>user\n"..query.."<|im_end|>\n<|im_start|>assistant\n<think>";
        if(!thinking) text=text.."</think>";
        if(!tokenizer.Encode(text,pids)) return false;
        maxTokens=max(0,CVar.GetCVar('llm_max_tokens').GetInt());
        if(pids.Size()<1) return false;
        if(debug || CVar.GetCVar('llm_tokenize_only').GetBool())
        {
            string line="PIDS";
            for(int i=0;i<pids.Size();i++) line=line..String.Format(" %d",pids[i]);
            Console.Printf("%s",line);
        }
        return true;
    }

    ui void Start()
    {
        if(mode!=0) { Console.Printf("LFM3 is working; use llm_cancel before starting another request."); return; }
        debug=CVar.GetCVar('llm_debug').GetBool();
        if(!Tokenize()) return;
        if(CVar.GetCVar('llm_tokenize_only').GetBool()) { Console.Printf("TOKENIZE_DONE"); return; }
        started=MSTimeF(); lastReport=started; computeMs=0; maxSlice=0; maxInferenceSlice=0;
        generated=0; position=0; reply=""; pendingUtf8="";
        renderCount=0; renderSlow=0; renderTotal=0; renderMax=0; renderWasRunning=false;
        if(model==null)
        {
            model=new("LFMModel");
            if(!model.Configure()) { model=null; Console.Printf("LLM_ERROR weights could not be configured"); return; }
        }
        model.debug=debug;
        model.profile=CVar.GetCVar('llm_profile').GetBool();
        loadActiveMs=0; setupMs=MSTimeF()-started;
        if(model.profile)
        {
            let jit=CVar.FindCVar('vm_jit');
            let aot=CVar.FindCVar('vm_jit_aot');
            Console.Printf("PROFILE_VM jit=%d aot=%d",jit==null ? -1 : jit.GetInt(),aot==null ? -1 : aot.GetInt());
        }
        mode=1; loadStarted=MSTimeF();
        if(maxTokens>0) Console.Printf("LFM3: %d prompt tokens; up to %d reply tokens. Use llm_cancel to stop.",pids.Size(),maxTokens);
        else Console.Printf("LFM3: %d prompt tokens; generating until end of reply or llm_cancel.",pids.Size());
    }

    ui string Escaped(string s)
    {
        string result;
        for(int i=0;i<int(s.Length());i++)
        {
            int c=s.ByteAt(i);
            if(c>=32 && c<127 && c!=92) result=result..s.Mid(i,1);
            else result=result..String.Format("\\x%02X",c);
        }
        return result;
    }

    ui void StreamToken(int tid)
    {
        string piece=tokenizer.Decode(tid);
        if(generated==0 && thinking) Console.Printf("LFM3 <think>");
        reply=reply..piece;
        pendingUtf8=pendingUtf8..piece;


        int at=0;
        while(at<int(pendingUtf8.Length()))
        {
            int c=pendingUtf8.ByteAt(at);
            int count=c<128 ? 1 : (c>=240 ? 4 : (c>=224 ? 3 : (c>=192 ? 2 : 1)));
            if(at+count>int(pendingUtf8.Length())) break;
            at+=count;
        }
        if(at>0)
        {
            Console.PrintfEx(PRINT_HIGH|PRINT_NONOTIFY,"%s",pendingUtf8.Left(at));
            pendingUtf8=pendingUtf8.Mid(at);
        }
    }

    ui bool AcceptToken(int tid)
    {
        if(debug) Console.Printf("TOK %d %d",generated,tid);
        if(tid!=124900) StreamToken(tid);
        generated++;
        if(tid==124900 || (maxTokens>0 && generated>=maxTokens))
        { Finish(); return true; }
        return false;
    }

    ui void Finish()
    {
        mode=0;
        if(debug || model.profile)
        {
            Console.Printf("TEXT %s",Escaped(reply));
            Console.Printf("RENDER samples=%d avg_ms=%.3f max_ms=%.3f over50ms=%d",renderCount,renderTotal/max(renderCount,1),renderMax,renderSlow);
            Console.Printf("PERF tokens=%d elapsed_ms=%.3f compute_ms=%.3f max_slice_ms=%.3f",generated,MSTimeF()-started,computeMs,maxSlice);
        }
        Console.Printf("LFM3: %s",reply);
        if(model.profile)
        {
            Console.Printf("PROFILE_REQUEST setup_ms=%.3f load_active_ms=%.3f load_wall_ms=%.3f inference_wall_ms=%.3f",setupMs,loadActiveMs,prefillStarted-loadStarted,MSTimeF()-prefillStarted);
            for(int i=0;i<model.stageMs.Size();i++)
                if(model.stageCalls[i]>0) Console.Printf("PROFILE_STAGE %d %.6f %d",i,model.stageMs[i],model.stageCalls[i]);
        }
        Console.Printf("LLM_DONE");
    }

    override void UiTick()
    {
        if(!autoChecked && gamestate==GS_LEVEL)
        {
            autoChecked=true;
            Console.Printf("LFM3 console: llm_prompt \"your question\" then llm. Use llm_cancel to stop.");
            if(CVar.GetCVar('llm_autorun').GetBool()) Start();
        }

        double budget=clamp(CVar.GetCVar('llm_budget_ms').GetFloat(),1.0,8.0);
        RunSlice(MSTimeF()-lastRender>50 ? min(8.0,budget*2.0) : budget);
    }

    override void RenderOverlay(RenderEvent e)
    {
        double now=MSTimeF();
        if(mode==2 && renderWasRunning)
        {
            double interval=now-lastRender;
            renderCount++; renderTotal+=interval; renderMax=max(renderMax,interval);
            if(interval>50) renderSlow++;
        }
        renderWasRunning=mode==2;
        lastRender=now;
        RunSlice(clamp(CVar.GetCVar('llm_budget_ms').GetFloat(),1.0,8.0));
    }

    ui void RunSlice(double budget)
    {
        if(mode==0) return;
        double timeout=CVar.GetCVar('llm_timeout_s').GetFloat();
        if(timeout>0 && MSTimeF()-started>timeout*1000.0)
        {
            mode=0;
            Console.Printf("LFM3: stopped after %.0f seconds. %s",timeout,reply);
            Console.Printf("LLM_TIMEOUT");
            return;
        }
        int initialMode=mode;

        double t=MSTimeF();
        while(mode!=0 && MSTimeF()-t<budget)
        {
            if(mode==1)
            {
                if(model.error) { mode=0; Console.Printf("LLM_ERROR loading failed"); break; }
                double loadT=model.profile ? MSTimeF() : 0;
                bool loaded=model.LoadStep();
                if(model.profile) loadActiveMs+=MSTimeF()-loadT;
                if(loaded)
                {
                    if(debug) Console.Printf("LOAD_MS %.3f",MSTimeF()-loadStarted);
                    prefillStarted=MSTimeF();
                    model.Reset(); model.StartToken(pids[0],0,pids.Size()==1); mode=2;
                }
            }
            else if(model.Step())
            {
                if(position<pids.Size()-1)
                {
                    position++;
                    model.StartToken(pids[position],position,position==pids.Size()-1);
                }
                else
                {
                    int tid=model.nextId;
                    if(generated==0 && debug) Console.Printf("LOGIT0 %d %.17g",tid,model.nextValue);
                    if(!AcceptToken(tid)) { position++; model.StartToken(tid,position,true); }
                }
            }
        }
        double elapsed=MSTimeF()-t;
        computeMs+=elapsed; maxSlice=max(maxSlice,elapsed);
        if(initialMode==2) maxInferenceSlice=max(maxInferenceSlice,elapsed);
        if(mode!=0 && MSTimeF()-lastReport>=5000)
        {
            if(mode==1) Console.Printf("LFM3 loading: %d/%d matrices",model.loadIndex,model.matrices.Size());
            else if(position<pids.Size())
            {
                double progress=position+min(model.layer,24)/24.0;
                double remaining=progress>0 ? (MSTimeF()-prefillStarted)/1000.0*(pids.Size()-progress)/progress : 0;
                if(progress>=1) Console.Printf("LFM3 reading: %d/%d tokens, about %.0f seconds to first reply token",position,pids.Size(),remaining);
                else Console.Printf("LFM3 reading: %d/%d tokens (layer %d/24)",position,pids.Size(),min(model.layer+1,24));
            }
            else Console.Printf("LFM3 writing: %d tokens",generated);
            lastReport=MSTimeF();
        }
    }
}
