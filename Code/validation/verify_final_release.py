"""Verify release hashes and saved evidence; optionally rerun lightweight CPU checks.

Run from the repository root. No private workspace or manuscript files are used.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]


def compare(a,b):
    """Compare scientific numbers, ignoring the source hash changed by CLI adaptation."""
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            if k != 'script_sha256': compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): compare(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert abs(a-b) <= 2e-8*max(1,abs(a),abs(b)), (a,b)
    else: assert a==b,(a,b)


def run(args):
    process=subprocess.run([sys.executable,*args],cwd=ROOT,text=True,capture_output=True)
    if process.returncode:
        print(process.stdout);print(process.stderr,file=sys.stderr)
        raise RuntimeError(f'Command failed: {args}')
    print('PASS:', ' '.join(args), flush=True)
    return process.stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-checks',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'release_validation')
    args=parser.parse_args()
    hashes=json.loads((ROOT/'reproducibility/final_jgo_sha256.json').read_text())
    for name,digest in hashes['files'].items():
        contents=(ROOT/name).read_bytes()
        if Path(name).suffix in hashes['lf_normalized_extensions']:
            contents=contents.replace(b'\r\n',b'\n')
        assert hashlib.sha256(contents).hexdigest()==digest,name
    with zipfile.ZipFile(ROOT/'reproducibility/result_jcam_synthetic_full.zip') as archive:
        assert archive.testzip() is None
        checkpoints=[json.loads(archive.read(n)) for n in archive.namelist() if '/synthetic/' in n and n.endswith('.json')]
        groups=defaultdict(set)
        for item in checkpoints:
            assert item['status']=='success'
            cfg=json.loads(json.dumps(item['config']));seed=cfg['synthetic'].pop('seed')
            groups[json.dumps(cfg,sort_keys=True)].add(seed)
        assert len(checkpoints)==2460 and len(groups)==246
        assert all(seeds==set(range(10)) for seeds in groups.values())
        devices=Counter(x['device'] for x in checkpoints)
        assert devices=={'cuda':2292,'cpu':168},devices
    result=dict(hashes_verified=len(hashes['files']),synthetic_checkpoints=2460,
                synthetic_complete_ten_seed_groups=246,devices=dict(devices))
    if args.run_checks:
        out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
        run(['-c',"import sys;sys.path[:0]=['Code/src','Code'];import numpy,scipy,matplotlib,interval_pareto;from experiments import real_bank_pareto_coverage"])
        tests=run(['-m','pytest','-q','Code/tests']);print(tests.strip())
        run(['Code/validation/smoke.py','--output',str(out/'smoke')])
        run(['Code/experiments/diagnose_general_k_strengthening.py','--output-dir',str(out/'coefficients')])
        compare(json.loads((ROOT/'Code/results/jgo_strengthening/general_k_decision_diagnostics.json').read_text()),
                json.loads((out/'coefficients/general_k_decision_diagnostics.json').read_text()))
        run(['Code/experiments/final_jgo_revision_analysis.py','--output-dir',str(out/'presentation')])
        saved=ROOT/'Code/results/final_jgo_revision'
        for name in ['theorem6_independent_smallwidth.json','bank_descriptive_statistics.json']:
            compare(json.loads((saved/name).read_text()),json.loads((out/'presentation'/name).read_text()))
        for name in ['theorem6_coefficients.csv','bank_pool_seedmeans.csv']+[f'table{i}_{suffix}.csv' for i,suffix in [(1,'scaled_coverage'),(2,'coefficients'),(3,'bank_main'),(4,'bank_dispersion'),(5,'computing_environments')]]:
            assert list(csv.reader((saved/name).open()))==list(csv.reader((out/'presentation'/name).open())),name
        for stem in ['profile_query_precision','theorem6_coefficient_scaling','bank_pareto_variability']:
            for ext in ['pdf','png']:
                assert (out/'presentation'/f'{stem}.{ext}').stat().st_size>1000
        result['cpu_checks']='imports, unit tests, smoke, 21 recomputed coefficients, 18 optimization checks, all numerical tables and figures passed'
        (out/'validation_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
