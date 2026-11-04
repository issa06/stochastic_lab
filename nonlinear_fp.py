"""Finite-amplitude periodic FP response from connected two-time covariances."""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded
from scipy.integrate import trapezoid
from fokker_planck import generator


def periodic_snr(model, D, A, bands, observable, grid, steps=128, phases=16,
                 lag_cycles=8, max_cycles=120, tolerance=1e-8):
    """Numerical finite-domain FP, without linearizing the external forcing.

    Implicit transition matrices propagate both probabilities and zero-mass
    conditional vectors. Phase-averaged connected covariance supplies the noise
    spectrum; the fundamental of the periodic mean supplies coherent power.
    Unresolved periodic states or covariance tails never return a valid SNR.
    """
    if any(not isinstance(v,(int,np.integer)) for v in (steps,phases,lag_cycles,max_cycles)):
        raise ValueError("非線形FPの分割・周期数は整数です。")
    if not np.isfinite(A) or A < 0 or bands.f0 <= 0:
        raise ValueError("外力振幅と周波数を確認してください。")
    if not 64 <= steps <= 2048 or not 4 <= phases <= 64 or phases > steps or not 4 <= lag_cycles <= 64:
        raise ValueError("非線形FPの時間分割・位相数・相関周期数を確認してください。")
    if steps*phases*lag_cycles*len(grid) > 100_000_000 or max_cycles < 3:
        raise ValueError("非線形FPの計算量が上限を超えています。")
    dt = 1/(bands.f0*steps)
    matrices = [generator(model,D,grid,A*np.cos(2*np.pi*(j+1)/steps)).implicit_matrix(dt) for j in range(steps)]
    p = generator(model,D,grid).probability.copy()
    mass_error, minimum, residual = 0., 1., np.inf
    for cycle in range(1,max_cycles+1):
        old = p.copy()
        frames = []
        for matrix in matrices:
            frames.append(p.copy())
            p = solve_banded((1,1),matrix,p,check_finite=False)
            mass_error = max(mass_error,abs(p.sum()-1))
            minimum = min(minimum,float(p.min()))
        residual = float(np.linalg.norm(p-old,1))
        if cycle >= 3 and residual < tolerance:
            break
    frames = np.asarray(frames)
    g = model.observable(np.asarray(grid),observable)
    mean = frames@g
    z = 2*np.mean(mean*np.exp(-2j*np.pi*np.arange(steps)/steps))
    signal = float(abs(z)**2/2)
    starts = np.floor(np.arange(phases)*steps/phases).astype(int)
    vectors = (frames[starts]*(g[None,:]-mean[starts,None])).T.copy()
    covariance = np.empty(steps*lag_cycles+1)
    covariance[0] = float(np.mean(np.sum((g[:,None]-mean[starts])*vectors,axis=0)))
    for lag in range(1,len(covariance)):
        for j, start in enumerate(starts):
            index = (start+lag-1) % steps
            vectors[:,j] = solve_banded((1,1),matrices[index],vectors[:,j],check_finite=False)
            # Remove floating-point stationary-mode contamination only.
            vectors[:,j] -= frames[(index+1)%steps]*vectors[:,j].sum()
        positions = (starts+lag) % steps
        covariance[lag] = float(np.mean(np.sum((g[:,None]-mean[positions])*vectors,axis=0)))
    lags = np.arange(len(covariance))*dt
    freq = bands.frequency[bands.noise]
    integrand = covariance[:,None]*np.cos(2*np.pi*lags[:,None]*freq)
    density = 4*trapezoid(integrand,x=lags,axis=0)
    halfway = len(lags)//2+1
    half_density = 4*trapezoid(integrand[:halfway],x=lags[:halfway],axis=0)
    change = float(np.max(abs(density-half_density))/max(float(np.max(abs(density))),1e-15))
    tail = float(np.max(abs(covariance[-steps:]))/max(abs(covariance[0]),1e-15))
    boundary = float(np.max(frames[:,:3].sum(axis=1)+frames[:,-3:].sum(axis=1)))
    periodic_ok = residual < tolerance and mass_error < 1e-7 and minimum >= -1e-10
    tail_ok = tail <= 1e-4 and change <= .02
    valid = bool(periodic_ok and tail_ok and boundary <= 1e-6 and np.all(density>0))
    noise = float(np.mean(density)*bands.bandwidth)
    return {"method":"FP nonlinear", "D":D, "approximation":"finite_amplitude_connected_covariance",
            "snr":signal/noise if valid and noise>0 else np.nan, "signal_power":signal,"noise_power":noise,
            "valid":valid,"periodic_converged":bool(periodic_ok),"tail_converged":bool(tail_ok),
            "periodic_residual":residual,"tail_ratio":tail,"spectrum_truncation_change":change,
            "mass_error":float(mass_error),"minimum_probability":minimum,"boundary_mass":boundary,
            "steps":steps,"phases":phases,"lag_cycles":lag_cycles,"cycles":cycle,"dt":dt,
            "lag":lags,"covariance":covariance,"noise_frequency":freq,"noise_psd":density,
            "time":np.arange(steps)*dt,"mean":mean,"grid":grid,"probability":frames}


def resolution_study(model,D,A,bands,observable,grid,**kwargs):
    base = periodic_snr(model,D,A,bands,observable,grid,**kwargs)
    checks = []
    from fokker_planck import make_grid
    dx = grid[1]-grid[0]
    domain = (grid[0]-dx/2,grid[-1]+dx/2)
    mid = (domain[0]+domain[1])/2
    radius = (domain[1]-domain[0])/2
    cases = [("時間刻み半減",grid,dict(kwargs,steps=2*base["steps"])),
             ("FP格子細分化",make_grid(domain,min(2001,2*len(grid)-1)),kwargs),
             ("領域拡大",make_grid((mid-1.25*radius,mid+1.25*radius),min(2001,int(np.ceil(1.25*len(grid))))),kwargs),
             ("位相数倍増",grid,dict(kwargs,phases=2*base["phases"]))]
    for name,points,settings in cases:
        try:
            r = periodic_snr(model,D,A,bands,observable,points,**settings)
            change = abs(r["snr"]-base["snr"])/max(abs(base["snr"]),1e-12)
            valid = base["valid"] and r["valid"]
            checks.append({"check":name,"snr":r["snr"],"relative_change":change,
                           "result":"PASS" if valid and change<=.05 else "FAIL" if valid else "NOT RUN"})
        except (ValueError,ArithmeticError):
            checks.append({"check":name,"result":"NOT RUN"})
    base["resolution_checks"] = checks
    base["resolution_supported"] = all(r["result"]=="PASS" for r in checks)
    return base
