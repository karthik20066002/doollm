#!/usr/bin/env python3
import json
import re
import tempfile
from pathlib import Path
from engine import run_engine
from common import BUILD, write_json


def main():
    with tempfile.TemporaryDirectory(prefix="lfm-profile-") as directory:
        tmp = Path(directory)
        (tmp / "MAPINFO").write_text(
            'GameInfo { AddEventHandlers = "LFMForwardProfile" }'
        )
        (tmp / "ZSCRIPT").write_text("""version "4.14"
class LFMForwardProfile : StaticEventHandler
{
    ui bool configured;
    override void UiTick()
    {
        if(configured) return;
        let runner=LFMRunner(StaticEventHandler.Find("LFMRunner"));
        if(runner==null || runner.mode!=1) return;
        runner.pids.Clear(); runner.pids.Push(124894);
        runner.maxTokens=1;
        configured=true;
        Console.Printf("PROFILE_INPUT raw_bos=124894 positions=1 outputs=1");
    }
}
""")
        log, metrics = run_engine(
            query="profile",
            system="",
            max_tokens=1,
            debug=False,
            profile=True,
            timeout=180,
            label="forward_profile",
            extra_files=[tmp],
        )
    stages = {
        int(stage): {"ms": float(ms), "calls": int(calls)}
        for stage, ms, calls in re.findall(
            r"^PROFILE_STAGE (\d+) (\S+) (\d+)$", log, re.M
        )
    }
    groups = {
        "ffn_matvec": [31, 32, 34],
        "operator_projections": [11, 13, 20, 21, 22, 23, 26],
        "vocabulary_matvec": [40],
        "attention": [25],
    }
    grouped = {
        name: sum(stages.get(i, {}).get("ms", 0) for i in ids)
        for name, ids in groups.items()
    }
    total = sum(s["ms"] for s in stages.values())
    grouped["other"] = total - sum(grouped.values())
    request = dict(
        (k, float(v))
        for k, v in re.findall(
            r"(\w+_ms)=(\S+)", re.search(r"^PROFILE_REQUEST (.+)$", log, re.M)[1]
        )
    )
    report = {
        "workload": "one raw BOS token through 24 layers and vocabulary projection; cold load",
        "vm": re.search(r"^PROFILE_VM (.+)$", log, re.M)[1],
        "stages": stages,
        "stage_total_ms": total,
        "groups": {
            k: {"ms": v, "percent": 100 * v / total} for k, v in grouped.items()
        },
        "request": request,
        **metrics,
    }
    write_json(BUILD / "forward_profile.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
