"""Small CPU diagnostics of fixed-K oracle geometry; no proof by numerics."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import random

R = 0.125

def bisect_inc(fn, target, lo, hi):
    for _ in range(56):
        mid = (lo + hi) / 2
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def profile(scale, beta):
    return (lambda z: scale * (z*z + beta*z**4),
            lambda z: scale * (2*z + 4*beta*z**3))

def run_case(k, params, rng):
    p, dp = profile(*params[0])
    q, dq = profile(*params[1])
    f = lambda x: p(x + R)
    g = lambda x: q(R - x)
    df = lambda x: dp(x + R)
    dg = lambda x: -dq(R - x)
    def greedy(e):
        left, xs = -R, []
        for _ in range(k):
            x = bisect_inc(f, min(f(R), f(left)+e), -R, R)
            xs.append(x)
            target = max(g(R), g(x)-e)
            left = bisect_inc(lambda z: -g(z), -target, -R, R)
        return xs, left
    low, high = 0., max(f(R), g(-R))
    for _ in range(58):
        mid = (low+high)/2
        xs, _ = greedy(mid)
        if g(xs[-1])-mid <= g(R)+1e-17:
            high = mid
        else:
            low = mid
    e = high
    xs, _ = greedy(e)
    def constraints(points):
        vals = [f(points[0])-f(-R)]
        crosses = []
        for a,b in zip(points, points[1:]):
            u = bisect_inc(lambda z: f(z)-g(z), f(b)-g(a), a,b)
            crosses.append(u)
            vals.append(g(a)-g(u))
        vals.append(g(points[-1])-g(R))
        return vals, crosses
    vals, us = constraints(xs)
    aa, bb = [], []
    for i,u in enumerate(us):
        den = df(u)-dg(u)
        aa.append(-dg(xs[i])*df(u)/den)
        bb.append(-dg(u)*df(xs[i+1])/den)
    upper = [1/df(xs[0])]
    for a,b in zip(aa,bb):
        upper.append((1+a*upper[-1])/b)
    lower = [0.]*k
    lower[-1] = 1/(-dg(xs[-1]))
    for i in range(k-2,-1,-1):
        lower[i] = (1+bb[i]*lower[i+1])/aa[i]
    sharp_c = max(upper+lower)
    alpha = 1/(2*sharp_c)
    # Exact tangent minimax LP: positive dual weights of the chain Jacobian.
    p0, qk = df(xs[0]), -dg(xs[-1])
    weights = [1.]
    if k == 1:
        weights.append(p0/qk)
    else:
        weights.append(p0/aa[0])
        for i in range(k-2):
            weights.append(weights[-1]*bb[i]/aa[i+1])
        weights.append(weights[-1]*bb[-1]/qk)
    total = sum(weights)
    weights = [w/total for w in weights]
    rhs = [p0]+[abs(b-a) for a,b in zip(aa,bb)]+[qk]
    rho = sum(w*b for w,b in zip(weights,rhs))
    mu1, mu2 = (2*p[0] for p in params)
    m1, m2 = (p[0]*(2+12*p[1]) for p in params)
    l_edge=max(dp(2*R),dq(2*R))
    h_max=dp(2*R)+dq(2*R)
    h_min=2*R*min(mu1,mu2)
    dd=(R/2)*(math.sqrt(m1*max(m1,m2))/mu1+math.sqrt(m2*max(m1,m2))/mu2)
    dd+=2*h_max*math.sqrt(m1*m2)/(mu1*mu2)
    beta=3*max(m1/mu1,m2/mu2)+l_edge/h_min
    rho_lower=2*e/dd
    rho_upper=beta*e/R
    assert rho_lower <= rho+1e-12 <= rho_upper+1e-12
    z = [(rho-p0)/p0]
    for a,b,shift in zip(aa,bb,rhs[1:-1]):
        z.append((rho-shift+a*z[-1])/b)
    tangent_residual = abs(-qk*z[-1]+qk-rho)
    assert tangent_residual < 1e-12
    local_width_checks=[]
    for s in [1e-4,1e-5,1e-6]:
        candidate=[x+s*v for x,v in zip(xs,z)]
        worst=max(max(constraints([x-s*t for x in candidate])[0])-e
                  for t in [-1+i/20 for i in range(41)])
        local_width_checks.append(dict(s=s, achieved_excess_over_s=worst/s,
                                       tangent_coefficient=rho))
    query_checks=0
    s=1e-5
    lip_full=max(dp(1.),dq(1.))
    for theta in [-s,0.,s]:
        for nu in [0.,1e-6]:
            actual=f(xs[-1]-theta)
            # Adversarial one-sided interval [actual, actual+2nu].
            mid=actual+nu
            projected=min(f(xs[-1]+s),max(f(xs[-1]-s),mid))
            estimate=bisect_inc(lambda t:-f(xs[-1]-t),-projected,-s,s)
            err=abs(estimate-theta)
            excess=max(constraints([x+estimate-theta for x in xs])[0])-e
            assert excess <= lip_full*err+1e-12
            if nu==0.:
                assert err<1e-12
            query_checks+=1
    local_min = math.inf
    # Deterministic seed, very small perturbations; not a global-gap test.
    for _ in range(120):
        v = [rng.uniform(-1e-7,1e-7) for _ in xs]
        yy = [x+d for x,d in zip(xs,v)]
        ratio = (max(constraints(yy)[0])-e)/max(map(abs,v))
        local_min = min(local_min,ratio)
    # Dense targets underestimate the exact root-based coverage.
    dense = max(min(max(f(x)-f(u),g(x)-g(u),0.) for x in xs)
                for u in [-R+2*R*i/1600 for i in range(1601)])
    m = min(dp(R/2),dq(R/2))
    lip = max(dp(2*R),dq(2*R))
    if k == 1 and params == ((1.,0.),(1.,0.)):
        assert abs(e-R*R)<1e-13
    if k == 2 and params == ((1.,0.),(1.,0.)):
        assert abs(e-9*R*R/16)<1e-13
        assert max(abs(x-y) for x,y in zip(xs,[-R/4,R/4]))<1e-12
    assert m*R/(2*k)-1e-13 <= e <= lip*R/k+1e-13
    assert max(abs(z-e) for z in vals)<1e-12
    assert dense <= max(vals)+1e-12
    return dict(K=k, profiles=params, floor=e, K_times_floor=k*e,
                representatives=xs, max_active_residual=max(abs(z-e) for z in vals),
                sharpness_C=sharp_c, candidate_local_alpha=alpha,
                first_order_minimax_coefficient=rho, K_times_coefficient=k*rho,
                proven_coefficient_lower=rho_lower, proven_coefficient_upper=rho_upper,
                tangent_optimal_shift=z, tangent_dual_weights=weights,
                tangent_equations_residual=tangent_residual,
                first_order_width_diagnostics=local_width_checks,
                one_query_upper_checks=query_checks,
                min_sampled_local_ratio=local_min,
                sampled_local_bound_holds=local_min>=alpha-1e-8,
                exact_minus_dense=max(vals)-dense,
                reflection_error=max(abs(x+y) for x,y in zip(xs,reversed(xs))))

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'jgo_strengthening')
    args=parser.parse_args()
    rng=random.Random(20261028)
    profiles=[((1.,0.),(1.,0.)),((1.,1.),(1.,1.)),((1.,0.),(3.,2.))]
    rows=[run_case(k,p,rng) for p in profiles for k in [1,2,3,4,8,16,32]]
    result=dict(seed=20261028, cases=len(rows), perturbations_per_case=120,
                numerical_results_are_proof=False, rows=rows,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=args.output_dir
    path.mkdir(parents=True,exist_ok=True)
    (path/'general_k_decision_diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(cases=len(rows),max_residual=max(r['max_active_residual'] for r in rows),
        min_sampled_ratio_over_alpha=min(r['min_sampled_local_ratio']/r['candidate_local_alpha'] for r in rows),
        all_sampled_local_checks=all(r['sampled_local_bound_holds'] for r in rows)),indent=2))

if __name__=='__main__':
    main()
