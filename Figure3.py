import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import cmasher as cmr
import optuna
import torch
import emulator_helpers as em
import tutorial
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from tqdm import tqdm
from scipy.stats import norm
from sklearn.metrics import r2_score

BW = False
if BW:
    mpl.rcParams['axes.facecolor']='k'
    mpl.rcParams['figure.facecolor']='k'
    mpl.rcParams['text.color']='w'
    mpl.rcParams['axes.edgecolor']='w'
    mpl.rcParams['xtick.color']='w'
    mpl.rcParams['ytick.color']='w'
    mpl.rcParams['axes.labelcolor'] = 'w'

mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.linewidth'] = 2.25*1.25
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5*1.25
mpl.rcParams['ytick.major.width'] = 1.5*1.25
mpl.rcParams['xtick.minor.width'] = 1.0*1.25
mpl.rcParams['ytick.minor.width'] = 1.0*1.25
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4.5
mpl.rcParams['ytick.minor.size'] = 4.5
mpl.rcParams['xtick.top']   = True
mpl.rcParams['ytick.right'] = True

halo_mass  = np.load('./data/halo_mass.npy')
def norm_params(params):
    nparams = params / np.array([1, 1, 3.6, 7.4, .1, 1.0])
    
    minimum = np.array([0.274, 0.780, 0.25, 0.5, 0.25, np.min(halo_mass)])
    maximum = np.array([0.354, 0.888,  4.0, 2.0, 4.0 , np.max(halo_mass)])

    nparams = (nparams - minimum)/(maximum - minimum)
    
    return nparams


param_path = '/standard/DREAMS/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'
sim_params = tutorial.get_params(param_path)
params     = tutorial.get_params(param_path)
params     = np.array(params)
sat_params = [] ## add in halo mass
for box in range(1024):
    c = list(params[box])
    c.append(halo_mass[box])
    sat_params.append(c)
sim_params = np.array(sat_params)

boxes      = np.arange(1024)
snap       = 90
h          = 0.6909
in_file    = './data/CDM_density_profiles.hdf5' ## precomputed density profiles

rvir = np.load('./data/rvir.npy') / h

best_params = []

with h5py.File('./data/gnfw.hdf5', 'r') as file:
    for box in boxes:
        best_params.append(list(file[f'box_{box:04d}']['params']))

best_params = np.array(best_params)
    
slopes = best_params[:,0:2]

cmap = cmr.get_sub_cmap('cmr.pepper', 0.1, 0.9, N=1024)

def get_emulator_param(which,npoints=1000):
    sn1     = np.ones(npoints)*3.6
    sn2     = np.ones(npoints)*7.4
    bhff    = np.ones(npoints)*0.1
    omega_m = np.ones(npoints)*0.31
    sigma_8 = np.ones(npoints)*0.8159
    mhalo   = np.ones(npoints)*12.0

    if which == 0:
        sn1 = np.linspace(0.25,4.0,npoints)*3.6
        var = sn1
    elif which == 1:
        sn2 = np.linspace(0.5,2.0,npoints)*7.4
        var = sn2
    elif which == 2:
        bhff = np.linspace(0.25,4.0,npoints)*0.1
        var = bhff
    elif which == 3:
        omega_m = np.linspace(0.274,0.354,npoints)
        var = omega_m
    elif which == 4:
        sigma_8 = np.linspace(0.780,0.888,npoints)
        var = sigma_8
    elif which == 5:
        mhalo = np.linspace(np.log10(5.8e11), np.log10(2e12), npoints)
        var = mhalo
    
    # Set up the emulator params (features to input into the model)
    emulator_params = np.array([omega_m, sigma_8, sn1, sn2, bhff, mhalo]).T
    x = norm_params(emulator_params)
    return x, var

## Get GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)


all_preds_sn1 = []
all_errs_sn1  = []
all_preds_sn2 = []
all_errs_sn2  = []
all_preds_agn = []
all_errs_agn  = []
all_preds_om  = []
all_errs_om   = []
all_preds_s8  = []
all_errs_s8   = []
all_preds_mh  = []
all_errs_mh   = []

sn1 = None
sn2 = None
agn = None
om  = None
s8  = None
mh  = None

s = []

def coverage_plot(y_true, y_pred, y_std, ax, color='k'):    
    sigmas = np.linspace(0,4,100)
    coverage = np.zeros(len(sigmas))
    
    for i, s in enumerate(sigmas):
        lower = y_pred - s*y_std
        upper = y_pred + s*y_std
        coverage[i] = np.mean((y_true >= lower) & (y_true <= upper))
    
    ax.plot(sigmas, coverage, color=color)
    
    ideal_coverage = 2*norm.cdf(sigmas) - 1
    ax.plot(sigmas, ideal_coverage, ls='--', color='k')
    
all_y = []
all_e = []

valid = np.arange(1024)

for i in range(10):
    name    = f"gNFW_norm"
    storage = f"sqlite:///{os.getcwd()}/Databases/optuna_{name}_{i:02d}"

    study = optuna.load_study(study_name=name, storage=storage)

    best_trial = study.best_trial
    params     = best_trial.params

    learning_rate = params["learning_rate"]
    weight_decay  = params["weight_decay"]
    n_layers      = params["n_layers"]
    out_features  = [params[f"n_units_l{i}"] for i in range(n_layers)]
    dropout_rate  = [params[f"dropout_l{i}"] for i in range(n_layers)]
    input_size    = 6
    output_size   = 2
    nepoch        = 500
    name_model    = name

    hparams = em.Hyperparameters(
        learning_rate,
        weight_decay,
        n_layers,
        out_features,
        dropout_rate,
        nepoch,
        output_size,
        name_model,
        input_size
    )

    model = em.load_model(hparams, device=device).to(device)

    ## validation plot
    # slice_ = 10
    
    x        = norm_params(np.array(sim_params[valid]))
    eparam   = torch.tensor(x, dtype=torch.float, device=device)
    out      = model(eparam).cpu().detach().numpy()
    means    = out[:, :output_size]
    raw_stds = out[:, output_size:]
    stds     = np.log1p(np.exp(raw_stds))
    
    mean = np.mean(slopes, axis=0)
    std = np.std(slopes, axis=0)
    out = (slopes - mean) / std

    y = means * std + mean
    e = stds * std
    
    all_y.append(y)
    all_e.append(e)
    
    for p in range(6):
        x, var = get_emulator_param(p)
        eparam = torch.tensor(x, dtype=torch.float, device=device)
                
        # Make predictions using the model
        out = model(eparam).cpu().detach().numpy()
        means = out[:, :output_size]
        raw_stds = out[:, output_size:]
        stds     = np.log1p(np.exp(raw_stds))
        
        mean = np.mean(slopes, axis=0)
        std = np.std(slopes, axis=0)
        out = (slopes - mean) / std
        
        y = means * std + mean
        e = stds * std

        # Plot for individual model (optional)
        if p in [0, 1, 2]:
            var = np.log10(var)
        
        # Store predictions and uncertainties
        if p == 0:
            all_preds_sn1.append(y)
            all_errs_sn1.append(e)
            sn1 = var
        elif p == 1:
            all_preds_sn2.append(y)
            all_errs_sn2.append(e)
            sn2 = var
        elif p == 2:
            all_preds_agn.append(y)
            all_errs_agn.append(e)
            agn = var
        elif p == 3:
            all_preds_om.append(y)
            all_errs_om.append(e)
            om = var
        elif p == 4:
            all_preds_s8.append(y)
            all_errs_s8.append(e)
            s8 = var
        elif p == 5:
            all_preds_mh.append(y)
            all_errs_mh.append(e)
            mh = var

all_y = np.array(all_y)
all_e = np.array(all_e)

all_preds_sn1 = np.array(all_preds_sn1)
all_errs_sn1  = np.array(all_errs_sn1 )
all_preds_sn2 = np.array(all_preds_sn2)
all_errs_sn2  = np.array(all_errs_sn2 )
all_preds_agn = np.array(all_preds_agn)
all_errs_agn  = np.array(all_errs_agn )
all_preds_om  = np.array(all_preds_om )
all_errs_om   = np.array(all_errs_om  )
all_preds_s8  = np.array(all_preds_s8 )
all_errs_s8   = np.array(all_errs_s8  )
all_preds_mh  = np.array(all_preds_mh )
all_errs_mh   = np.array(all_errs_mh  )


fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
# labels = [r'$\log(A_{\rm SN1})$', r'$\log(A_{\rm SN2})$', r'$\log(A_{\rm AGN})$',
#           r'$\Omega_{\rm M}$', r'$\sigma_8$']
labels = [r'$\log(\bar{e}_w)$', r'$\log(\kappa_w)$', r'$\log(\epsilon_{f,\,{\rm high}})$',
          r'$\Omega_{\rm M}$', r'$\sigma_8$', r'$\log(M_{\rm Halo}~[{\rm M}_\odot])$']
axs = axs.flatten()

colors  = [cmap(0.66), cmap(0.66), cmap(0.66)]
markers = ['s','o','d']
ls      = ['-',':','-.']

all_preds = [all_preds_sn1, all_preds_sn2, all_preds_agn, all_preds_om, all_preds_s8, all_preds_mh]
all_errs  = [all_errs_sn1, all_errs_sn2, all_errs_agn, all_errs_om, all_errs_s8, all_errs_mh]
all_vars  = [sn1, sn2, agn, om, s8, mh]
names     = ['sn1', 'sn2', 'agn', 'omega_m', 'sigma_8', 'mhalo']
fids      = [np.log10(3.6), np.log10(7.4), np.log10(0.1), 0.31, 0.8159, 12.0]
txt_locs  = [np.log10(3.8), np.log10(7.7), np.log10(0.105), 0.3125, 0.82, 12.05]

sim_params = [sim_params[:,2], sim_params[:,3], sim_params[:,4], sim_params[:,0], sim_params[:,1], halo_mass]

for i, this_pred in enumerate(all_preds):
    this_err  = all_errs[i]
    this_var  = all_vars[i]
    ax        = axs[i]
    this_sp   = np.log10(sim_params[i]) if i < 3 else sim_params[i]
    this_name = names[i]

    ensemble_mean = np.mean(this_pred, axis=0)
    ensemble_std  = np.std(this_pred, axis=0)
    model_uncertainty = np.mean(np.abs(this_err), axis=0)
    total_uncertainty = np.sqrt(ensemble_std**2 + np.mean(this_err**2, axis=0))
        
    print(np.mean(total_uncertainty,axis=0))
        
    for j in range(2):        
        if j == 0:
            ax.plot(this_var, ensemble_mean[:, j], color=colors[j], lw=3, ls=ls[j])            
            ax.fill_between(this_var,
                            (ensemble_mean[:, j] + total_uncertainty[:, j]),
                            (ensemble_mean[:, j] - total_uncertainty[:, j]),
                            color=colors[j], alpha=0.35)
            
            ax.plot(this_var,ensemble_mean[:, j] + total_uncertainty[:, j], color=colors[j], lw=1)
            ax.plot(this_var,ensemble_mean[:, j] - total_uncertainty[:, j], color=colors[j], lw=1)
                        
            ax.scatter(this_sp, slopes[:, j], color=colors[j], marker=markers[j], s=10, alpha=0.1,
                       rasterized=True, facecolor='none')

            validation_x    = np.linspace(np.min(this_sp), np.max(this_sp), 40)
            dx = validation_x[1] - validation_x[0]
            
            validation_x    = np.linspace(np.min(this_sp), np.max(this_sp)+dx, 40)
            validation_y    = np.zeros(len(validation_x))
            validation_yerr = np.zeros(len(validation_x))
            
            for index, x in enumerate(validation_x):
                within_dx = (this_sp > x) & (this_sp < x+dx)
                validation_y[index]    = np.mean(slopes[:, j] [within_dx])
                validation_yerr[index] = np.std(slopes[:, j] [within_dx])
            
            ax.plot(validation_x, validation_y, color='k', lw=3)
            ax.fill_between(validation_x, 
                            validation_y+validation_yerr,
                            validation_y-validation_yerr,
                            color='k', alpha=0.3
            )
            
    ax.set_xlabel(labels[i])
        
    ax.axvline(x=fids[i],ymin=0.9,ymax=1, color='red', ls='-', alpha=0.5,lw=2)
    if i != 5:
        ax.text(txt_locs[i],7.32,r'${\rm TNG~Fiducial}$',color='red',fontsize=12,ha='left',va='bottom', alpha=0.5)
    
    if i in [0, 3]:
        ax.set_ylabel(r'$\log(\rho_s~[M_\odot/{\rm kpc}^3])$')
        # ax.set_ylabel(r'$\log(\rho_s~[{\rm kpc}])$')

# axs[-1].axis('off')
        
plt.tight_layout()
plt.subplots_adjust(wspace=0.025, hspace=0.3)
plt.savefig('./figs/Figure3.pdf', bbox_inches='tight')
