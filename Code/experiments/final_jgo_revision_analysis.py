"""Reproduce final presentation from immutable saved records; CPU only.

The nonlinear diagnostic optimizes a deterministic, sampled-world problem.
It is not a certified computation of randomized continuum-world R_K(s).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
from scipy.optimize import brentq, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Code/results/final_jgo_revision'
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
                     'axes.spines.right':False})
COLORS=['#174a7e','#a34a13','#376d35']
STYLES=['o-','s--','^-.']
LABELS=['A: quadratic / quadratic','B: quartic / quartic','C: quadratic / unequal quartic']

def digest(p):
    # Git checkouts may use LF or CRLF; text provenance is canonicalized to LF.
    return hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def readcsv(p):
    return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(p.open())]
def savefig(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)

def coefficients():
    source=ROOT/'Code/results/jgo_strengthening/general_k_decision_diagnostics.json'
    data=json.loads(source.read_text())
    assert data['cases']==21 and len(data['rows'])==21
    pairs=[]
    for row in data['rows']:
        if row['profiles'] not in pairs: pairs.append(row['profiles'])
    assert len(pairs)==3
    flat=[]
    fig,ax=plt.subplots(figsize=(6.0,3.6))
    for n,pair in enumerate(pairs):
        rows=sorted([r for r in data['rows'] if r['profiles']==pair],key=lambda r:r['K'])
        assert [r['K'] for r in rows]==[1,2,3,4,8,16,32]
        for r in rows:
            rho=r['first_order_minimax_coefficient']
            assert r['proven_coefficient_lower']<=rho<=r['proven_coefficient_upper']
            flat.append(dict(profile=chr(65+n),K=r['K'],rho=rho,K_rho=r['K']*rho,
                lower=r['proven_coefficient_lower'],upper=r['proven_coefficient_upper']))
        xx=[r['K'] for r in rows]; yy=[r['first_order_minimax_coefficient'] for r in rows]
        ax.loglog(xx,yy,STYLES[n],color=COLORS[n],label=LABELS[n],markersize=4)
    ax.loglog([1,32],[.25,.25/32],':',color='black',linewidth=1,
              label=r'$0.25/K$ comparison (not fitted)')
    ax.set_xticks([1,2,4,8,16,32],labels=['1','2','4','8','16','32'])
    ax.set(xlabel=r'Output budget $K$',ylabel=r'First-order coefficient $\rho_K$')
    ax.grid(True,which='major',alpha=.22)
    ax.legend(frameon=False)
    fig.tight_layout();savefig(fig,'theorem6_coefficient_scaling')
    with (OUT/'theorem6_coefficients.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    return data,dict(coefficient_input_sha256=digest(source),configurations=21)

def independent_diagnostic(data):
    """Independent root evaluator + nonlinear epigraph optimizer, no tangent z initialization."""
    rows=[]
    rng=np.random.default_rng(20261028)
    for rec in data['rows']:
        k=rec['K']
        if k not in (1,2,3): continue
        (s1,b1),(s2,b2)=rec['profiles']
        f=lambda y:s1*((y+.125)**2+b1*(y+.125)**4)
        g=lambda y:s2*((.125-y)**2+b2*(.125-y)**4)
        ystar=np.array(rec['representatives']);e=rec['floor']
        def active(y):
            vals=[f(y[0])-f(-.125)]
            for a,b in zip(y[:-1],y[1:]):
                u=brentq(lambda u:f(u)-g(u)-f(b)+g(a),a,b,xtol=5e-15)
                vals.append(g(a)-g(u))
            vals.append(g(y[-1])-g(.125))
            return np.array(vals)
        for width in (1e-4,1e-5):
            worlds=np.linspace(-1,1,33)
            def losses(z):
                return np.concatenate([(active(ystar+width*(z-t))-e)/width for t in worlds])
            results=[]
            for initial in [np.zeros(k),rng.uniform(-.2,.2,k)]:
                x0=np.r_[initial,max(losses(initial))+.001]
                opt=minimize(lambda z:z[-1],x0,method='SLSQP',
                    bounds=[(-3.,3.)]*k+[(0.,None)],
                    constraints=[{'type':'ineq','fun':lambda z:z[-1]-losses(z[:-1])}],
                    options={'maxiter':160,'ftol':1e-9})
                actual=max(losses(opt.x[:-1]))
                if opt.success and actual<=opt.x[-1]+2e-6:
                    results.append((actual,opt))
            assert results, (k,rec['profiles'],width,'no converged start')
            value,opt=min(results,key=lambda t:t[0])
            row=dict(K=k,profiles=rec['profiles'],s=width,world_count=33,
                optimized_sampled_world_excess_over_s=float(value),
                rho_formula=rec['first_order_minimax_coefficient'],
                relative_difference=float(value/rec['first_order_minimax_coefficient']-1),
                successful_starts=len(results),method='SLSQP exact-target epigraph; two starts',
                certification='Neither global optimization nor randomized continuum minimax certified')
            rows.append(row)
    (OUT/'theorem6_independent_smallwidth.json').write_text(json.dumps(rows,indent=2))
    return dict(cases=len(rows),max_abs_relative_difference=max(abs(r['relative_difference']) for r in rows))

def bank():
    source=ROOT/'Code/results/real_bank_pareto_coverage/bank_pareto_coverage_rounds.csv'
    records=readcsv(source)
    assert len(records)==1920
    fractions=sorted(set(r['sample_fraction'] for r in records))
    batches=sorted(set(r['batch'] for r in records));seeds=sorted(set(r['seed'] for r in records))
    assert len(batches)==128 and len(seeds)==5 and fractions==[.5,.8,.95]
    assert len({(r['batch'],r['seed'],r['sample_fraction']) for r in records})==1920
    report=[];poolrows=[]
    for frac in fractions:
        subset=[r for r in records if r['sample_fraction']==frac]
        matrix=np.array([[next(r['adaptive_excess'] for r in subset if r['batch']==b and r['seed']==s)
                          for s in seeds] for b in batches])
        static=np.array([next(r['blind_excess'] for r in subset if r['batch']==b) for b in batches])
        for b in batches:
            rr=[r for r in subset if r['batch']==b]
            assert len({r['blind_excess'] for r in rr})==1
            for r in rr:
                assert r['interval_rate_lower']<=r['population_positive_rate']<=r['interval_rate_upper']
        poolmeans=matrix.mean(axis=1);seedmeans=matrix.mean(axis=0)
        gain=static-poolmeans
        item=dict(fraction=frac,records=len(subset),pools=128,seeds=5,
            static_mean=float(static.mean()),adaptive_mean=float(matrix.mean()),
            pool_mean_sd=float(poolmeans.std(ddof=1)),
            within_pool_seed_rms_sd=float(np.sqrt(np.mean(matrix.var(axis=1,ddof=1)))),
            seedwise_mean_sd=float(seedmeans.std(ddof=1)),seedwise_mean_min=float(seedmeans.min()),
            seedwise_mean_max=float(seedmeans.max()),paired_gain_mean=float(gain.mean()),
            paired_gain_pool_sd=float(gain.std(ddof=1)),
            fraction_record_improved=float(np.mean([r['adaptive_better'] for r in subset])),
            fraction_poolmean_improved=float(np.mean(gain>1e-12)),
            labels_read_mean=float(np.mean([r['sample_count'] for r in subset])),
            seedwise_means=seedmeans.tolist(),pool_quartiles=np.quantile(poolmeans,[.25,.5,.75]).tolist())
        report.append(item)
        for b,sm,am,gg in zip(batches,static,poolmeans,gain):
            poolrows.append(dict(fraction=frac,batch=int(b),static_excess=float(sm),
                                 adaptive_seedmean=float(am),paired_gain=float(gg)))
    summary=dict(input_sha256=digest(source),records=1920,pools=128,seeds=5,
                 interpretation='Descriptive dispersion on fixed chronological pools; no population CI',by_fraction=report)
    (OUT/'bank_descriptive_statistics.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'bank_pool_seedmeans.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(poolrows[0]));w.writeheader();w.writerows(poolrows)
    xx=np.array([r['fraction']*100 for r in report]); yy=np.array([r['adaptive_mean'] for r in report])
    lo=np.array([r['seedwise_mean_min'] for r in report]);hi=np.array([r['seedwise_mean_max'] for r in report])
    fig,ax=plt.subplots(figsize=(5.7,3.5))
    ax.plot(xx,[r['static_mean'] for r in report],'s--',color='#555555',label='Calibration-prefix pair')
    ax.errorbar(xx,yy,yerr=np.array([yy-lo,hi-yy]),fmt='o-',color='#174a7e',capsize=5,
                label='Adaptive mean; range of five seed means')
    for j,seed in enumerate(seeds):
        ax.scatter(xx,[r['seedwise_means'][j] for r in report],s=12,color='#174a7e',alpha=.45)
    ax.set_yscale('log');ax.set_xticks(xx)
    ax.set(xlabel='Age 40+ pool labels observed (%)',ylabel='Mean worst-target excess coverage')
    ax.grid(axis='y',alpha=.25);ax.legend(frameon=False,loc='best')
    fig.tight_layout();savefig(fig,'bank_pareto_variability')
    return summary

def profile_plot():
    source=ROOT/'Code/results/profile_class_feedback/profile_class_feedback.csv'
    rows=[r for r in readcsv(source) if r['beta']==1.]
    widths=sorted(set(r['s'] for r in rows))
    fig,ax=plt.subplots(figsize=(6.,3.5))
    ax.plot(widths,[next(r['worst_blind_excess'] for r in rows if r['s']==s) for s in widths],
            's--',color='#555555',label='Common envelopes: static pair')
    for j,factor in enumerate((0.,.1,.5)):
        yy=[next(r['worst_feedback_excess'] for r in rows if r['s']==s and r['nu_over_m_s_s']==factor) for s in widths]
        ax.plot(widths,yy,STYLES[j],color=COLORS[j],markersize=4,
                label=rf'One interval: $\nu/(m_s s)={factor:g}$')
    ax.set(xlabel=r'Hidden-world radius $s$',ylabel='Worst-world cumulative excess coverage')
    ax.grid(alpha=.22);ax.legend(frameon=False)
    fig.tight_layout();savefig(fig,'profile_query_precision')
    return digest(source)

def write_table(name, rows):
    with (OUT/(name+'.csv')).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    keys=list(rows[0])
    lines=['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']
    lines += ['| '+' | '.join(str(r[k]) for k in keys)+' |' for r in rows]
    (OUT/(name+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def tables(bank_report):
    source=ROOT/'Code/results/thm06_pareto_information/pareto_information_summary.json'
    raw=json.loads(source.read_text())['rows']
    write_table('table1_scaled_coverage',[
        dict(amplitude=r['amplitude'],K=r['K'],mean_blind_coverage=r['mean_blind_coverage'],
             oracle_grid_coverage=r['oracle_grid_coverage'],
             measured_excess_over_bound=r['mean_excess_over_oracle_grid']/r['theorem_excess_lower_bound'])
        for r in raw if r['amplitude']==1.])
    coeff=list(csv.DictReader((OUT/'theorem6_coefficients.csv').open()))
    write_table('table2_coefficients',coeff)
    rows=bank_report['by_fraction']
    write_table('table3_bank_main',[
        dict(fraction=r['fraction'],labels_read=r['labels_read_mean'],static_excess=r['static_mean'],
             adaptive_excess=r['adaptive_mean'],adaptive_over_static=r['adaptive_mean']/r['static_mean'],
             fraction_cases_improved=r['fraction_record_improved']) for r in rows])
    names=['fraction','adaptive_mean','pool_mean_sd','within_pool_seed_rms_sd','paired_gain_mean','paired_gain_pool_sd']
    write_table('table4_bank_dispersion',[{k:r[k] for k in names} for r in rows])
    env=json.loads((ROOT/'Code/configs/final_jgo_environments.json').read_text())
    write_table('table5_computing_environments',env['environments'])


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output_dir.resolve(); OUT.mkdir(parents=True,exist_ok=True)
    data,audit=coefficients();audit['independent_diagnostic']=independent_diagnostic(data)
    audit['bank']=bank();audit['profile_input_sha256']=profile_plot()
    audit['environment']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,device='CPU')
    tables(audit['bank'])
    audit['script_sha256']=digest(Path(__file__))
    (OUT/'presentation_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(dict(coefficients=audit['configurations'],independent=audit['independent_diagnostic'],
        bank=audit['bank']['by_fraction'],environment=audit['environment']),indent=2))

if __name__=='__main__':main()
