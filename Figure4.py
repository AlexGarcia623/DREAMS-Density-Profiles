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
mpl.rcParams['font.size'] = 20
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

with h5py.File('gnfw.hdf5', 'r') as file:
    for box in boxes:
        best_params.append(list(file[f'box_{box:04d}']['params']))
    
best_params = np.array(best_params)
    
slopes = best_params[:,2:5]

print(np.mean(slopes,axis=0))

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

sn1 = None

s = []

for i in range(10):
    name    = f"gNFW"
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
    output_size   = 3
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
    
    x, var = get_emulator_param(0)
    eparam = torch.tensor(x, dtype=torch.float, device=device)

    # Make predictions using the model
    out      = model(eparam).cpu().detach().numpy()
    means    = out[:, :output_size]
    raw_stds = out[:, output_size:]
    stds     = np.log1p(np.exp(raw_stds))

    mean = np.mean(slopes, axis=0)
    std = np.std(slopes, axis=0)
    out = (slopes - mean) / std

    y = means * std + mean
    e = stds * std

    # Plot for individual model (optional)
    var = np.log10(var)

    # Store predictions and uncertainties
    all_preds_sn1.append(y)
    all_errs_sn1.append(e)
    sn1 = var

all_preds_sn1 = np.array(all_preds_sn1)
all_errs_sn1  = np.array(all_errs_sn1 )

# fig = plt.figure(figsize=(9,4)) ## Validation Plot
fig = plt.figure(figsize=(7,5.5))
ax  = plt.gca()

labels = [r'$\log(\bar{e}_w)$']

colors  = [cmap(0.15), cmap(0.45), cmap(0.75)]
markers = ['s','o','d']
ls      = ['--','-','-.']

all_preds = [all_preds_sn1]
all_errs  = [all_errs_sn1]
all_vars  = [sn1]
names     = ['sn1']
fids      = [np.log10(3.6)]

sim_params = [sim_params[:,2], sim_params[:,3], sim_params[:,4], sim_params[:,0], sim_params[:,1]]

for i, this_pred in enumerate(all_preds):
    this_err  = all_errs[i]
    this_var  = all_vars[i]
    this_sp   = np.log10(sim_params[i]) if i < 3 else sim_params[i]
    this_name = names[i]

    ensemble_mean = np.mean(this_pred, axis=0)
    ensemble_std  = np.std(this_pred, axis=0)
    model_uncertainty = np.mean(this_err, axis=0)
    total_uncertainty = np.sqrt(ensemble_std**2 + np.mean(this_err**2, axis=0))
    
    for j in range(3):
        ax.plot(this_var, ensemble_mean[:, j], color=colors[j], lw=3, ls=ls[j])        
        ax.fill_between(this_var,
                        (ensemble_mean[:, j] + total_uncertainty[:, j]),
                        (ensemble_mean[:, j] - total_uncertainty[:, j]),
                        color=colors[j], alpha=0.35)
        
        ax.plot(this_var,ensemble_mean[:, j] + total_uncertainty[:, j],color=colors[j], lw=1)
        ax.plot(this_var,ensemble_mean[:, j] - total_uncertainty[:, j],color=colors[j], lw=1)
        
#         ax.scatter(this_sp, slopes[:, j], color=colors[j], marker=markers[j], s=10, alpha=0.3,
#                    rasterized=True, facecolor='none')
    
#         validation_x    = np.linspace(np.min(this_sp), np.max(this_sp), 40)
#         dx = validation_x[1] - validation_x[0]

#         validation_x    = np.linspace(np.min(this_sp), np.max(this_sp)+dx, 40)
#         validation_y    = np.zeros(len(validation_x))
#         validation_yerr = np.zeros(len(validation_x))

#         dx = validation_x[1] - validation_x[0]

#         for index, x in enumerate(validation_x):
#             within_dx = (this_sp > x) & (this_sp < x+dx)
#             validation_y[index]    = np.mean(slopes[:, j] [within_dx])
#             validation_yerr[index] = np.std(slopes[:, j] [within_dx])

#         ax.plot(validation_x, validation_y, color='k', lw=3, ls='-')
#         ax.fill_between(validation_x, 
#                         validation_y+validation_yerr,
#                         validation_y-validation_yerr,
#                         color='k', alpha=0.25
#         )
        
    ax.axhline(1.0, color='gray', ls=':', alpha=0.5, lw=3)
    ax.axhline(3.0, color='gray', ls=':', alpha=0.5, lw=3)
    if i == 0:
        ax.text(-0.05,1.05,r'${\rm NFW~}\alpha,\ \gamma$', color='gray')
        ax.text(-0.05,3.05,r'${\rm NFW~}\beta$', color='gray')

    ax.set_xlabel(labels[i])
    
    ax.axvline(x=fids[i],ymin=0.9,ymax=1, color='red', ls='-', alpha=0.5,lw=2)
    ax.text(np.log10(3.8),3.1,r'${\rm TNG~Fiducial}$',color='red',
            fontsize=15,ha='left',va='bottom', alpha=0.5)
    
    if i in [0, 3]:
        ax.set_ylabel(r'${\rm Shape~Parameter}$')

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax*1.1)

legend_elements = [
    #Line2D([0], [0], color='k', lw=3, ls='-', label=r'${\rm Raw~Simulation~Mean}$'), ## Validation
    Line2D([0], [0], color=colors[1], lw=3, ls=ls[1], label=r'$\beta~({\rm Outer~Slope})$'),
    Line2D([0], [0], color=colors[2], lw=3, ls=ls[2], label=r'$\gamma~({\rm Inner~Slope})$'),
    Line2D([0], [0], color=colors[0], lw=3, ls=ls[0], label=r'$\alpha~({\rm Transition})$'),
]

# leg = ax.legend(frameon=False,handles=legend_elements, bbox_to_anchor=(1,1), fontsize=14)
leg = ax.legend(frameon=False,handles=legend_elements, loc='center right', fontsize=14)

# colors = ['k',colors[1],colors[2], colors[0]]
colors = [colors[1],colors[2], colors[0]]
for index, text in enumerate(leg.get_texts()):
    text.set_color(colors[index])
        
plt.tight_layout()
# plt.subplots_adjust(wspace=0.025, hspace=0.3)
plt.savefig('./figs/Figure5_only_sn1.pdf', bbox_inches='tight')
