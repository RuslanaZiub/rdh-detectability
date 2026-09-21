from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, platform, sys, subprocess
import numpy, pandas, PIL, sklearn, skimage, scipy


def capture_environment(results_root: str | Path = '/workspace/results') -> Path:
    root=Path(results_root); root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=root/f'run_{stamp}'; run.mkdir(parents=True,exist_ok=False)
    def cmd(args):
        try: return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
        except Exception as e: return f'unavailable: {e}'
    obj={
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),
        'machine':platform.machine(),
        'processor':platform.processor(),
        'python':sys.version,
        'packages':{
            'numpy':numpy.__version__,'pandas':pandas.__version__,'Pillow':PIL.__version__,
            'scikit-learn':sklearn.__version__,'scikit-image':skimage.__version__,'scipy':scipy.__version__},
        'git_commit':cmd(['git','rev-parse','HEAD']),
    }
    (run/'environment.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    (root/'LATEST_RUN.txt').write_text(run.name,encoding='utf-8')
    return run
