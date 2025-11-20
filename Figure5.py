import sys, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import cmasher as cmr
import optuna
import torch
import emulator_helpers as em
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

import tutorial

BW = False
if BW: ## if black background
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

##############################
### Load Simulation Params ###
##############################

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

######################################################
### Load in data *exactly* as you did for training ###
######################################################

boxes      = np.arange(1024)
snap       = 90
h          = 0.6909
in_file    = './data/CDM_baryon_contract_update.hdf5'

slopes     = np.zeros(len(boxes))
rvir       = np.load('./data/rvir.npy') / h
rvir_dmo   = np.load('../CDM_DMO/data/rvir.npy')

cmap = cmr.get_sub_cmap('cmr.pepper', 0.1, 0.9, N=1024)

with h5py.File(in_file, 'r') as file:
    for index, box in enumerate(np.arange(0,1024)):
        if box in [846,796]: ## DNE
            slopes[box] = np.nan
            continue
        if box in [10,15,448,476,503,656,704]: ## select wrong subhalo
            slopes[box] = np.nan
            continue
        if box in [410,547,581,781,808,829,984]: ## select wrong halo
            slopes[box] = np.nan
            continue
        this_box = file[f'box_{box:04d}']
        radius   = np.array(this_box['radius'])
        hydro_dm = np.array(this_box['menc_hydro_dm'])
        dmo_dm   = np.array(this_box['menc_dmo'])
                
        hydro_dm = 10**hydro_dm[radius < 275]
        dmo_dm   = 10**dmo_dm  [radius < 275]
        radius   = radius[radius < 275]
        
        r_hydro = radius/rvir[box]
        r_dmo   = radius/rvir_dmo[box]
        
        slopes[box] = np.log10(hydro_dm[np.argmin(np.abs(r_hydro-0.01))]) - np.log10(dmo_dm[np.argmin(np.abs(r_dmo-0.01))])
        
no_nans = ~np.isnan(slopes)
slopes  = slopes[no_nans]
sim_params = sim_params[no_nans, :]
######################################################

    
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

## Get GPU if available
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

sn1 = None
sn2 = None
agn = None
om  = None
s8  = None

for i in range(10): ## for i in range(n_models_you_trained)
    name    = f"mass_growth" ### name of model you ran
    storage = f"sqlite:///{os.getcwd()}/Databases/optuna_{name}_{i:02d}"

    study = optuna.load_study(study_name=name, storage=storage)

    best_trial = study.best_trial
    params     = best_trial.params

    learning_rate = params["learning_rate"]
    weight_decay  = params["weight_decay"]
    n_layers      = params["n_layers"]
    out_features  = [params[f"n_units_l{i}"] for i in range(n_layers)]
    dropout_rate  = [params[f"dropout_l{i}"] for i in range(n_layers)]
    
    ### i/o
    input_size    = 6
    output_size   = 1
    nepoch        = 100  ## not sure it matters if you get this right or not?
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

    for p in range(5):
        x, var = get_emulator_param(p)
        eparam = torch.tensor(x, dtype=torch.float, device=device)
                
        ## Make predictions using the model
        pred = model(eparam).cpu().detach().numpy()
        out = model(eparam).cpu().detach().numpy()
        means    = out[:, 0]
        raw_std  = out[:, 1]
        pred_std = np.log1p(np.exp(raw_std))

        mean = np.mean(slopes, axis=0)
        std = np.std(slopes, axis=0)
        out = (slopes - mean) / std
        
        y = means * std + mean
        e = pred_std * std

        ## Get astroparams in log
        if p in [0, 1, 2]:
            var = np.log10(var)
        
        ## Store predictions and uncertainties
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

#################
### Make Plot ###
#################

fig, axs = plt.subplots(2,3, figsize=(13,7), sharey=True)
axs = axs.flatten()

# labels = [r'$\log(A_{\rm SN1})$',r'$\log(A_{\rm SN2})$',r'$\log(A_{\rm AGN})$',
#           r'$\Omega_{\rm M}$', r'$\sigma_8$']
labels = [r'$\log(\bar{e}_w)$', r'$\log(\kappa_w)$', r'$\log(\epsilon_{f,\,{\rm high}})$',
          r'$\Omega_{\rm M}$', r'$\sigma_8$']
fids      = [np.log10(3.6), np.log10(7.4), np.log10(0.1), 0.31, 0.8159] ## fiducial values
txt_locs  = [np.log10(3.8), np.log10(7.6), np.log10(0.105), 0.306, 0.811] ## text locations for fiducial values

all_preds = [
    all_preds_sn1, all_preds_sn2, all_preds_agn,
    all_preds_om , all_preds_s8
]
all_errs = [
    all_errs_sn1, all_errs_sn2, all_errs_agn,
    all_errs_om , all_errs_s8
]
all_vars = [
    sn1, sn2, agn, om, s8
]

names = [
    'sn1', 'sn2', 'agn', 'omega_m', 'sigma_8'
]

for i, this_pred in enumerate(all_preds):
    this_err  = all_errs[i]
    this_var  = all_vars[i]
    ax        = axs[i]
    this_sp   = np.log10(sim_params[i]) if i < 3 else sim_params[i]
    this_name = names[i]
    print(this_name)
    
    # Compute ensemble mean and standard deviation
    ensemble_mean = np.mean(this_pred, axis=0)
    ensemble_std  = np.std(this_pred, axis=0)

    # Compute the total uncertainty (aleatoric + epistemic)
    model_uncertainty = np.mean(this_err, axis=0)
    total_uncertainty = np.sqrt(ensemble_std**2 + np.mean(this_err**2, axis=0))
    
    print(np.mean(total_uncertainty))
    
    # Plot the final ensemble prediction with uncertainty
    ax.plot(this_var, ensemble_mean, color='k', lw=4.5)
    ax.plot(this_var, ensemble_mean, color=cmap(0.6) if BW else cmap(0.25), lw=3)
    ax.fill_between(this_var, (ensemble_mean + total_uncertainty), (ensemble_mean - total_uncertainty),
                    color=cmap(0.6) if BW else cmap(0.25), alpha=0.5)
    ax.plot(this_var, (ensemble_mean + total_uncertainty), color=cmap(0.6) if BW else cmap(0.25), lw=1)
    ax.plot(this_var, (ensemble_mean - total_uncertainty), color=cmap(0.6) if BW else cmap(0.25), lw=1)
    
#     validation_x    = np.linspace(np.min(this_sp), np.max(this_sp), 40)
#     dx = validation_x[1] - validation_x[0]

#     validation_x    = np.linspace(np.min(this_sp), np.max(this_sp)+dx, 40)
#     validation_y    = np.zeros(len(validation_x))
#     validation_yerr = np.zeros(len(validation_x))

#     for index, x in enumerate(validation_x):
#         within_dx = (this_sp > x) & (this_sp < x+dx)
#         validation_y[index]    = np.mean(slopes[within_dx])
#         validation_yerr[index] = np.std(slopes[within_dx])

#     ax.plot(validation_x, validation_y, color='k', lw=3)
#     ax.fill_between(validation_x, 
#                     validation_y+validation_yerr,
#                     validation_y-validation_yerr,
#                     color='k', alpha=0.3
#     )

    ax.set_xlabel(labels[i])

    ax.axhline(0.0, color='gray', ls='--', lw=2, alpha=0.5, zorder=-1)
    
    ax.axvline(x=fids[i],ymin=0.9,ymax=1, color='red', ls='-', alpha=0.5,lw=2)
    ax.text(txt_locs[i],0.65,r'${\rm TNG~Fiducial}$',color='red',fontsize=12,ha='left',va='bottom', alpha=0.5)

axs[0].text(-0.05,0.025,r'${\rm No~Change}$', color='gray', fontsize=14)
axs[0].set_ylabel(r'$\Gamma_{0.01}$')
axs[3].set_ylabel(r'$\Gamma_{0.01}$')

plt.tight_layout()
plt.subplots_adjust(wspace=0.025, hspace=0.3)
plt.savefig('./figs/Figure5.pdf', bbox_inches='tight')