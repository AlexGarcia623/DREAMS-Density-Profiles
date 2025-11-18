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

param_path = '/standard/DREAMS/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'
sim_params = tutorial.get_params(param_path)

boxes      = np.arange(1024)
snap       = 90
h          = 0.6909
in_file    = 'CDM_density_profiles.hdf5' ## precomputed density profiles

rvir = np.load('./data/rvir.npy') / h

best_params = []

stellar_mass = np.load('./data/stellar_mass.npy')
bh_mass      = np.load('./data/bh_mass.npy')

slopes   = np.zeros((len(boxes),2))
slopes_b = np.zeros((len(boxes),2))

for box in boxes:
    slopes[box,0] = stellar_mass[box]
    slopes[box,1] = np.log10(sim_params[box,4] * 0.1 * (10**stellar_mass[box])*2e30 * (3e8**2)) ## nonsense

    slopes_b[box,0] = bh_mass[box]
    slopes_b[box,1] = np.log10(sim_params[box,4] * 0.1 * (10**bh_mass[box])*2e30 * (3e8**2))
    
mask   = stellar_mass > 0
slopes = slopes[mask,:]

mask_b   = bh_mass > 0
slopes_b = slopes_b[mask_b,:]

sim_params = [sim_params[:,2], sim_params[:,3], sim_params[:,4], sim_params[:,0], sim_params[:,1]]

cmap = cmr.get_sub_cmap('cmr.pepper', 0.1, 0.9, N=1024)

def get_emulator_param(which,npoints=1000):
    sn1     = np.ones(npoints)*3.6
    sn2     = np.ones(npoints)*7.4
    bhff    = np.ones(npoints)*0.1
    omega_m = np.ones(npoints)*0.31
    sigma_8 = np.ones(npoints)*0.8159

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
    
    # Set up the emulator params (features to input into the model)
    emulator_params = np.array([omega_m, sigma_8, sn1, sn2, bhff]).T
    x = em.norm_params(emulator_params)
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

sn1 = None
sn2 = None
agn = None
om  = None
s8  = None

s = []

for i in range(10):
    name    = f"sh_mass"
    storage = f"sqlite:///{os.getcwd()}/Databases/optuna_{name}_{i:02d}"

    study = optuna.load_study(study_name=name, storage=storage)

    best_trial = study.best_trial
    params     = best_trial.params

    learning_rate = params["learning_rate"]
    weight_decay  = params["weight_decay"]
    n_layers      = params["n_layers"]
    out_features  = [params[f"n_units_l{i}"] for i in range(n_layers)]
    dropout_rate  = [params[f"dropout_l{i}"] for i in range(n_layers)]
    input_size    = 5
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

    for p in [0, 1]:
        x, var = get_emulator_param(p)
        eparam = torch.tensor(x, dtype=torch.float, device=device)
                
        # Make predictions using the model
        out = model(eparam).cpu().detach().numpy()
        means = out[:, :output_size]
        stds  = out[:, output_size:]
        
        mean = np.mean(slopes, axis=0)
        std = np.std(slopes, axis=0)
        out = (slopes - mean) / std
        
        y = means * std + mean
        e = stds * std
        # Denormalize the predictions
#         out  = np.log10(slopes + 1)
#         mean = np.mean(out, axis=0)
#         std  = np.std(out, axis=0)

#         y    = np.power(10, pred[:, 0] * std + mean) - 1  # Predicted values
#         e    = np.log(10) * std * np.power(10, pred[:, 0] * std + mean) * pred[:, 1]  # Uncertainty (error)

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

for i in range(10):
    name    = f"bh_mass"
    storage = f"sqlite:///{os.getcwd()}/Databases/optuna_bh_mass_{i:02d}"

    study = optuna.load_study(study_name=name, storage=storage)

    best_trial = study.best_trial
    params     = best_trial.params

    learning_rate = params["learning_rate"]
    weight_decay  = params["weight_decay"]
    n_layers      = params["n_layers"]
    out_features  = [params[f"n_units_l{i}"] for i in range(n_layers)]
    dropout_rate  = [params[f"dropout_l{i}"] for i in range(n_layers)]
    input_size    = 5
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

    for p in [2]:
        x, var = get_emulator_param(p)
        eparam = torch.tensor(x, dtype=torch.float, device=device)
                
        # Make predictions using the model
        out = model(eparam).cpu().detach().numpy()
        means = out[:, :output_size]
        stds  = out[:, output_size:]
        
        mean = np.mean(slopes_b, axis=0)
        std = np.std(slopes_b, axis=0)
        out = (slopes_b - mean) / std
                
        y = means * std + mean
        e = stds * std

        # Plot for individual model (optional)
        if p in [0, 1, 2]:
            var = np.log10(var)
        
        all_preds_agn.append(y)
        all_errs_agn.append(e)
        agn = var
            
all_preds_sn1 = np.array(all_preds_sn1)
all_errs_sn1  = np.array(all_errs_sn1 )
all_preds_sn2 = np.array(all_preds_sn2)
all_errs_sn2  = np.array(all_errs_sn2 )
all_preds_agn = np.array(all_preds_agn)
all_errs_agn  = np.array(all_errs_agn )

fig, axs = plt.subplots(1, 4, figsize=(13, 4), gridspec_kw={'width_ratios':[1,1,0.2,1]})
axs[2].axis('off')
axs = [axs[0], axs[1], axs[3]]
labels = [r'$\log(\bar{e}_w)$', r'$\log(\kappa_w)$', r'$\log(\epsilon_{f,\,{\rm high}})$']

colors  = [cmap(0.45), cmap(0.45), cmap(0.01), cmap(0.75)]
markers = ['s','o','d']
ls      = ['-',':','-.']

all_preds = [all_preds_sn1, all_preds_sn2, all_preds_agn]
all_errs  = [all_errs_sn1, all_errs_sn2, all_errs_agn]
all_vars  = [sn1, sn2, agn, om, s8]
names     = ['sn1', 'sn2', 'agn']
fids      = [np.log10(3.6), np.log10(7.4), np.log10(0.1)]
txt_locs  = [np.log10(3.8), np.log10(7.7), np.log10(0.105)]

for i, this_pred in enumerate(all_preds):
    this_err  = all_errs[i]
    this_var  = all_vars[i]
    ax        = axs[i]
    this_sp   = np.log10(sim_params[i]) if i < 3 else sim_params[i]
    this_sp   = this_sp[mask]
    this_name = names[i]

    ensemble_mean = np.mean(this_pred, axis=0)
    ensemble_std  = np.std(this_pred, axis=0)
    model_uncertainty = np.mean(this_err, axis=0)
    total_uncertainty = np.sqrt(ensemble_std**2 + np.mean(this_err**2, axis=0))
    
    ax.plot(this_var, ensemble_mean[:, 0], color=colors[i], lw=3, ls='-')
    ax.fill_between(this_var,
                    (ensemble_mean[:, 0] + total_uncertainty[:, 0]),
                    (ensemble_mean[:, 0] - total_uncertainty[:, 0]),
                    color=colors[i], alpha=0.35)
    ax.plot(this_var,ensemble_mean[:, 0] + total_uncertainty[:, 0],color=colors[i], lw=1)
    ax.plot(this_var,ensemble_mean[:, 0] - total_uncertainty[:, 0],color=colors[i], lw=1)
    
    
    ax.set_xlabel(labels[i])
        
    ax.axvline(x=fids[i],ymin=0.9,ymax=1, color='tomato' if BW else 'red', ls='-', alpha=0.5,lw=2)
    
    yval = 10.9
    if i == 2:
        yval = 8.09
    ax.text(txt_locs[i],yval,r'${\rm TNG~Fiducial}$',color='red',
            fontsize=12,ha='left',va='bottom', alpha=0.5)

axs[0].set_ylabel(r'$\log(M_\star~[M_\odot])$')
axs[2].set_ylabel(r'$\log(M_{\rm BH}~[M_\odot])$')

gbl_ymin, gbl_ymax = np.inf, -np.inf

for i in [0,1]:
    ymin, ymax = axs[i].get_ylim()
    if ymin < gbl_ymin:
        gbl_ymin = ymin
    if ymax > gbl_ymax:
        gbl_ymax = ymax
        
axs[0].set_ylim(gbl_ymin, gbl_ymax)
axs[1].set_ylim(gbl_ymin, gbl_ymax)

axs[1].set_yticklabels([])
        
# axs[0].text(0.05,0.075,r'${\rm SN~Wind~Energy}$',transform=axs[0].transAxes)
# axs[1].text(0.05,0.075,r'${\rm SN~Wind~Speeds}$',transform=axs[1].transAxes)
# axs[2].text(0.05,0.075,r'${\rm AGN~Thermal~Energy}$',transform=axs[2].transAxes)
    
plt.tight_layout()
plt.subplots_adjust(wspace=0.025, hspace=0.225)
plt.savefig('./figs/astro_variations.pdf', bbox_inches='tight')
