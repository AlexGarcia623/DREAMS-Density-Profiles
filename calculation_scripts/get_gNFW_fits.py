import sys
import emcee
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import corner
from mpi4py import MPI

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def gnfw(log_r, rho0, rs, alpha, beta, gamma):
    r    = 10**log_r
    rho0 = 10**rho0
    rs   = 10**rs
    x    = (r / rs)    
    return np.log10(rho0 / ( x**gamma * ( 1 + (x)**alpha )**((beta-gamma)/alpha) ))
    
def gnfw_r2(log_r, rho0, rs, alpha, beta, gamma):
    r    = 10**log_r
    rho0 = 10**rho0
    rs   = 10**rs
    x    = (r / rs)
    
    log_rho = np.log10(rho0 / ( x**gamma * (1 + x**alpha )**((beta - gamma)/alpha) ))
    return log_rho
    
def nfw(log_r, log_rho0, rs):
    r    = 10**log_r
    rho0 = 10**log_rho0
    rs   = 10**rs
    x    = (r / rs)
    return np.log10(rho0 / (x * (1 + x)**2))
    
# in_file = '../CDM_DMO/CDM_DMO_density_profiles.hdf5' ## precomputed density profiles

in_file = 'CDM_density_profiles.hdf5' ## precomputed density profiles

boxes = np.arange(1024)
rvir  = np.load('./data/rvir.npy') / 0.6909

out_file_name = f'./data/gnfw_{rank}.hdf5'

with h5py.File(out_file_name, 'w') as f:
    1==1

# for box in boxes:
def process_box(box):
    print(f'Starting box {box} on rank {rank}', flush=True)
    density = None
    radius  = None
    with h5py.File(in_file, 'r') as f:
        this_box = f[f'box_{box:04d}']
        density  = np.array(this_box['density'])
        radius   = np.array(this_box['radius'])
        spread   = np.array(this_box['spread'])
    
    this_rvir = rvir[box]
    
    low  = this_rvir*0.01
    high = this_rvir*1.0
    within_dr = (radius < high) & (radius > low)
    
    x    = np.log10(radius[within_dr])
    y    = np.log10(density[within_dr])
    yerr = spread[within_dr]
    # yerr = np.ones_like(y) * 0.1
    
    params_nfw, cov_nfw = curve_fit(nfw, x, y)
    pred_nfw = nfw(x, *params_nfw)
    perr_nfw = np.sqrt(np.diag(cov_nfw))
    
    def gnfw_wrapper(log_r, alpha, beta, gamma):
        return gnfw_r2(log_r, params_nfw[0], params_nfw[1], alpha, beta, gamma)
    
    p0  = [1.0, 3.0, 1.0]
    lower_bounds = [0.0, 1.5, 0.5]
    upper_bounds = [2.0, 4.0, 2.5]
    
    params, cov = curve_fit(gnfw_wrapper, x, y, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=3000)
    pred = gnfw_wrapper(x, *params)
    perr = np.sqrt(np.diag(cov))
    
    print(params_nfw, perr_nfw)
    print(params, perr)
    
    scale_uncert = 5
    
    if np.average(perr/params) > 5: ## if we're really not converged, just take the fiducial gNFW fit
        scale_uncert = np.average(params/perr)*2
    
    print(scale_uncert)
    def log_prior(theta):
        log_rho0, log_rs, alpha, beta, gamma = theta
        
        if not (0 < alpha and 0 < beta and 0 < gamma):
            return -np.inf
        
        if not (0 < log_rs and 0 < log_rho0):
            return -np.inf
        
        # Gaussian priors from NFW fit
        lp_rho0 = -0.5 * ((log_rho0 - params_nfw[0]) / (scale_uncert*perr_nfw[0]))**2
        lp_rs   = -0.5 * ((log_rs   - params_nfw[1]) / (scale_uncert*perr_nfw[1]))**2

        # Gaussian priors from gNFW shape fit
        lp_alpha = -0.5 * ((alpha - params[0]) / (scale_uncert*perr[0]))**2
        lp_beta  = -0.5 * ((beta  - params[1]) / (scale_uncert*perr[1]))**2
        lp_gamma = -0.5 * ((gamma - params[2]) / (scale_uncert*perr[2]))**2

        return lp_rho0 + lp_rs + lp_alpha + lp_beta + lp_gamma
    
    def log_likelihood(theta, x, y, yerr):
        model = gnfw(x, *theta)
        return -0.5 * np.sum(((y - model) / yerr)**2)
    # def log_likelihood(theta, x, y, yerr):
    #     return 0.0  # flat likelihood
    def log_posterior(theta, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)

    p0       = [params_nfw[0], params_nfw[1], params[0], params[1], params[2]]
    ndim     = len(p0)
    nwalkers = 64
    pos      = p0 + 1e-4 * np.random.randn(nwalkers, ndim)
    nsteps   = 10000
        
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, yerr))
    sampler.run_mcmc(pos, nsteps, progress=False)

    discard = nsteps // 3
    samples = sampler.get_chain(discard=discard, flat=True)

    best_params = np.median(samples, axis=0)
    spread      = np.std(samples, axis=0)
    
    print(best_params)
        
    with h5py.File(out_file_name, 'r+') as f:
        this_box = f.create_group(f'box_{box:04d}')
        this_box.create_dataset('params', data=best_params)
        this_box.create_dataset('uncert', data=spread)
        
    def sample_priors(n_samples=5000):
        log_rho0 = np.random.normal(loc=params_nfw[0],scale=scale_uncert*perr_nfw[0], size=n_samples)
        log_rs   = np.random.normal(loc=params_nfw[1],scale=scale_uncert*perr_nfw[1], size=n_samples)
        alpha    = np.random.normal(loc=params[0],scale=scale_uncert*perr[0],    size=n_samples)
        beta     = np.random.normal(loc=params[1],scale=scale_uncert*perr[1],    size=n_samples)
        gamma    = np.random.normal(loc=params[2],scale=scale_uncert*perr[2],    size=n_samples)
        return np.vstack([log_rho0, log_rs, alpha, beta, gamma]).T
    
    x = np.log10(radius[within_dr])
    y = np.log10(density[within_dr])
    pred = gnfw(x, *best_params)
    
    labels = ["rho0", "rs", "alpha", "beta", "gamma"]
    fig=corner.corner(samples, labels=labels)
    for ax in fig.get_axes():
        if ax.has_data():
            for coll in ax.collections:
                coll.set_rasterized(True)
    
    prior_samples = sample_priors(n_samples=5000)         
    corner.corner(
        prior_samples,
        labels=labels,
        color="C1",   # priors in orange
        fig=fig,
        hist_kwargs={"density": True, "linestyle": "--", "linewidth": 2},
        plot_datapoints=False,
        plot_contours=True,
        plot_density=True,
    )
    plt.savefig(f"./figs/corner/corner_{box}.pdf")
    plt.close()
    
    fig = plt.figure(figsize=(5,5))
    plt.plot(x, y, color='k', lw=3, alpha=0.5)
    plt.plot(x, pred, color='red', lw=3, alpha=0.5)
    
    plt.plot(x, gnfw_wrapper(x, *params), color='blue', lw=3, alpha=0.5)
    plt.plot(x, nfw(x, *params_nfw), color='green', lw=3, alpha=0.5)
    
    plt.savefig(f'./figs/corner/profile_{box}.pdf')
    plt.close()
        
boxes = np.arange(0, 1024)
boxes_per_rank = np.array_split(boxes, size)

for box in boxes_per_rank[rank]:
    process_box(box)