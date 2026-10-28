"""Check saved evidence independently of the presentation generator."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'Code/results'


def test_all_coefficient_cases_and_dual_residuals():
    data = json.loads((RESULTS/'jgo_strengthening/general_k_decision_diagnostics.json').read_text())
    assert data['seed'] == 20261028
    assert len(data['rows']) == data['cases'] == 21
    assert len({(str(r['profiles']),r['K']) for r in data['rows']}) == 21
    for row in data['rows']:
        assert row['K'] in (1,2,3,4,8,16,32)
        assert row['proven_coefficient_lower'] <= row['first_order_minimax_coefficient'] <= row['proven_coefficient_upper']
        assert row['max_active_residual'] < 1e-12
        assert row['tangent_equations_residual'] < 1e-12
        assert row['one_query_upper_checks'] == 6


def test_independent_diagnostics_discrepancy():
    rows=json.loads((RESULTS/'final_jgo_revision/theorem6_independent_smallwidth.json').read_text())
    assert len(rows) == 18
    assert len({(str(r['profiles']),r['K'],r['s']) for r in rows}) == 18
    differences=[]
    for row in rows:
        difference=row['optimized_sampled_world_excess_over_s']/row['rho_formula']-1
        assert abs(difference-row['relative_difference']) < 1e-12
        assert row['successful_starts'] == 2 and row['world_count'] == 33
        differences.append(abs(difference))
    assert abs(max(differences)-0.0004242618222640715) < 1e-9
    assert max(differences)*100 < .043


def test_bank_records_containment_and_paired_statistics():
    path=RESULTS/'real_bank_pareto_coverage/bank_pareto_coverage_rounds.csv'
    rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(path.open())]
    summary=json.loads((RESULTS/'final_jgo_revision/bank_descriptive_statistics.json').read_text())
    assert len(rows) == summary['records'] == 1920
    assert hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest() == summary['input_sha256']
    assert len({(r['batch'],r['seed'],r['sample_fraction']) for r in rows}) == 1920
    for r in rows:
        assert r['interval_rate_lower'] <= r['population_positive_rate'] <= r['interval_rate_upper']
    for reported in summary['by_fraction']:
        selected=[r for r in rows if r['sample_fraction']==reported['fraction']]
        batches=sorted({r['batch'] for r in selected})
        seeds=sorted({r['seed'] for r in selected})
        assert len(batches)==128 and len(seeds)==5
        matrix=np.array([[next(r['adaptive_excess'] for r in selected if r['batch']==b and r['seed']==s)
                          for s in seeds] for b in batches])
        static=np.array([next(r['blind_excess'] for r in selected if r['batch']==b) for b in batches])
        gains=static-matrix.mean(axis=1)
        assert abs(matrix.mean()-reported['adaptive_mean']) < 1e-12
        assert abs(gains.mean()-reported['paired_gain_mean']) < 1e-12
        assert abs(gains.std(ddof=1)-reported['paired_gain_pool_sd']) < 1e-12
        assert abs(np.sqrt(matrix.var(axis=1,ddof=1).mean())-reported['within_pool_seed_rms_sd']) < 1e-12
