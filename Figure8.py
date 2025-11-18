import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import h5py
import cmasher as cmr
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from tutorial import get_params

import matplotlib.gridspec as gridspec

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase

def get_bin_idx(point, npoints=29):
    om = np.linspace(0.274, 0.354, npoints)
    s8 = np.linspace(0.780, 0.888, npoints)
    sn1 = 10**np.linspace(np.log10(0.9), np.log10(14.4), npoints)
    sn2 = 10**np.linspace(np.log10(3.7), np.log10(14.8), npoints)
    agn = 10**np.linspace(np.log10(0.025), np.log10(0.4), npoints)

    bins = [om, s8, sn1, sn2, agn]
    bins = [sn1, sn2, agn]
    all_idx = []
    for i in range(len(bins)):
        idx = np.argmin(np.abs((bins[i] - point[i])))
        all_idx.append(idx)
    return np.array(all_idx)

mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.linewidth'] = 2.25*1
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5*1
mpl.rcParams['ytick.major.width'] = 1.5*1
mpl.rcParams['xtick.minor.width'] = 1.0*1
mpl.rcParams['ytick.minor.width'] = 1.0*1
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4.5
mpl.rcParams['ytick.minor.size'] = 4.5
mpl.rcParams['xtick.top']   = True
mpl.rcParams['ytick.right'] = True

in_file  = 'CDM_baryon_contract_update.hdf5'

basedir    = '/standard/DREAMS/'
sim_params = get_params(basedir+'Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt')

weights = np.load('./data/arya_weights_binned.npy')
params = sim_params[:,2:5]
ws = np.zeros(1024)
for i in range(1024):
    idx   = get_bin_idx(params[i]) 
    ws[i] = weights[idx[0]][idx[1]][idx[2]]

remove_idx = [846,796, 10,15,448,476,503,656,704,
              410,547,581,781,808,829,984, 745]

ws_filtered = np.delete(ws, remove_idx)

two_sigma_val = np.percentile(ws,25)

sim_params = [
    np.log10(sim_params[:,2]), np.log10(sim_params[:,3]), np.log10(sim_params[:,4]), 
    sim_params[:,0], sim_params[:,1]
]
# ps = [r'$A_{\rm SN1}$',r'$A_{\rm SN2}$',r'$A_{\rm AGN}$',r'$\Omega_{\rm M}$',r'$\sigma_8$']
ps    = [r'$\bar{e}_w$', r'$\kappa_w$', r'$\epsilon_{f,\,{\rm high}}$', r'$\Omega_{\rm M}$', r'$\sigma_8$']
def weighted_std(values, weights):
    average  = np.average(values, weights=weights, axis=0)
    return np.sqrt(np.average((values-average)**2, weights=weights, axis=0))

def plot_avgs(ax,all_profs,color,ls='-'):
    median  = np.median(all_profs, axis=0)
    low     = np.percentile(all_profs, 25, axis=0)
    high    = np.percentile(all_profs, 75, axis=0)
    ax.plot(x, median, color=color, lw=2.75,ls=ls)
    ax.fill_between(x, low, high, color=color, alpha=0.33)

class HandlerLineWithFill(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        fill_color, line_color = orig_handle
        # Create a rectangle (fill)
        fill = Rectangle([xdescent, ydescent], width, height,
                         facecolor=fill_color, edgecolor='none', alpha=0.3, transform=trans)
        # Create a line over the rectangle
        line = Line2D([xdescent, xdescent + width],
                      [ydescent + height / 2, ydescent + height / 2],
                      color=line_color, lw=2, transform=trans)
        return [fill, line]
    
h = 0.6909

rvir = np.load('./data/rvir.npy') / h

cmap = cmr.get_sub_cmap('cmr.iceburn', 0.2, 0.8, N=1024)

fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 0.5, 1, 1, 1])

ax_big = fig.add_subplot(gs[0:2, 0:2])

ax1 = fig.add_subplot(gs[0, 2])
ax2 = fig.add_subplot(gs[1, 2])
ax3 = fig.add_subplot(gs[0, 3])
ax4 = fig.add_subplot(gs[0, 4])
ax5 = fig.add_subplot(gs[0, 5])
ax6 = fig.add_subplot(gs[1, 3])
ax7 = fig.add_subplot(gs[1, 4])
ax8 = fig.add_subplot(gs[1, 5])

ax1.axis('off')
ax2.axis('off')
ax8.axis('off')

axs = [ax3, ax4, ax5, ax6, ax7]

for which_param in range(5):
    all_profs_low  = []
    all_profs_high = []
    all_profs_med  = []
    all_profs      = []
    
    dr         = 10 ## percent
    min_param  = np.percentile(sim_params[which_param],dr)
    max_param  = np.percentile(sim_params[which_param],100-dr)
    avg1_param = np.percentile(sim_params[which_param],50-dr/2)
    avg2_param = np.percentile(sim_params[which_param],50+dr/2)    

    with h5py.File(in_file, 'r') as file:
        for index, box in enumerate(np.arange(0,1024)):
            if box in [846,796]: ## DNE
                continue
            if box in [10,15,448,476,503,656,704]: ## select wrong subhalo
                continue
            if box in [410,547,581,781,808,829,984]: ## select wrong halo
                continue
                
            if box in [745]:
                continue
                
            this_param = sim_params[which_param][box]
            lower  = this_param <= min_param
            upper  = this_param >= max_param
            middle = (this_param > avg1_param) & (this_param <= avg2_param)
            plot   = lower or upper or middle
            
            this_box   = file[f'box_{box:04d}']
            radius     = np.array(this_box['radius'])
            hydro_dm   = np.array(this_box['menc_hydro_dm'])
            gnedin_dm  = np.array(this_box['menc_dmo_AC_pred'])

            hydro_dm  = hydro_dm 
            gnedin_dm = gnedin_dm
            radius    = radius   

            ratio = 10**gnedin_dm / 10**(hydro_dm)

            r = radius/rvir[box]

            ratio_prof = interp1d(r, ratio, fill_value='extrapolate')
            x = 10**(np.linspace(-2.2, 0, 100))
            
            if plot:
                if lower:
                    all_profs_low.append(ratio_prof(x))
                elif upper:
                    all_profs_high.append(ratio_prof(x))
                else:
                    all_profs_med.append(ratio_prof(x))
                    
            if which_param == 0:
                all_profs.append(ratio_prof(x))
                
                ax_big.plot(r, ratio, color='k', lw=1, alpha=0.01, rasterized=True)

    all_profs_low  = np.array(all_profs_low )
    all_profs_high = np.array(all_profs_high)
    all_profs_med  = np.array(all_profs_med )
    all_profs      = np.array(all_profs)

    plot_avgs(axs[which_param], all_profs_low , color=cmap(0.00), ls='--')
    plot_avgs(axs[which_param], all_profs_high, color=cmap(0.99), ls=':')
    
    axs[which_param].set_xscale('log')
    
    xmin, _ = axs[which_param].get_xlim()
    axs[which_param].set_xlim(xmin,10**(-0.9))
    
    axs[which_param].axhline(1.0, color='gray', ls='--', lw=3, alpha=0.5, zorder=-1)
    
    axs[which_param].text(0.95,0.925,ps[which_param], color='k',
                          transform=axs[which_param].transAxes, ha='right', va='top')
    
    if which_param == 0:
        median  = np.average(all_profs, weights=ws_filtered, axis=0)
        low     = median-weighted_std(all_profs, weights=ws_filtered)
        high    = median+weighted_std(all_profs, weights=ws_filtered)
        scatter = np.log10(high) - np.log10(low)
        ax_big.plot(x, median, color=cmap(0.75), lw=2.75)
        ax_big.fill_between(x, low, high, color=cmap(0.75), alpha=0.25)
        
        np.save('./data/F6_median_x.npy', x)
        np.save('./data/F6_median_y.npy', median)
        np.save('./data/F6_low_y.npy'   , low)
        np.save('./data/F6_high_y.npy'  , high)
        
        ax_big.set_xscale('log')
        
        legend_elements = [
            Line2D([0], [0], color='k', lw=2, alpha=0.25, ls='-', label=r'${\rm Idv.~ Halos}$'),
            (cmap(0.75), cmap(0.75)),
        ]

        legend_labels = [
            r'${\rm Idv.~ Halos}$',
            r'${\rm \hat{\mu}+\hat{\sigma}}$',
        ]
        leg = ax_big.legend(legend_elements, legend_labels, loc='upper right', frameon=False,
                            fontsize=12,
                            handler_map={tuple: HandlerLineWithFill()})
        
        colors = ['gray',cmap(0.75)]
        for index, text in enumerate(leg.get_texts()):
            text.set_color(colors[index])
        
        ax_big.axhline(1.0, color='gray', ls='--', lw=3, alpha=0.5, zorder=-1)
    
title = {
    0:r"$A_{\rm SN1}$",
    1:r"$A_{\rm SN2}$",
    2:r"$A_{\rm AGN}$",
    3:r"$\Omega_{\rm M}$",
    4:r"$\sigma_8$",
}

# plt.text(0.95,0.9,title[which_param],transform=plt.gca().transAxes,ha='right',fontsize=24)
ax8.plot([0,0],[1,1], alpha=1.0, label=r'${\rm Upper~%s}$'%dr + r'$\%$', ls=':',
                 color=cmap(0.99), lw=2.75)
ax8.plot([0,0],[1,1], alpha=1.0, label=r'${\rm Lower~%s}$'%dr + r'$\%$', ls='--',
         color=cmap(0.01), lw=2.75)

leg = ax8.legend(frameon=False, loc='center', fontsize=14,
                 labelspacing=0.05,handletextpad=0.1)
colors = [cmap(0.99), cmap(0.01)]
for index, text in enumerate(leg.get_texts()):
    text.set_color(colors[index])

# plt.xscale('log')

gbl_ymin = np.inf
gbl_ymax = -np.inf
for ax in [*axs]:
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    
    if ymin < gbl_ymin:
        gbl_ymin = ymin
    if ymax > gbl_ymax:
        gbl_ymax = ymax
        
ax_big.set_ylim(gbl_ymin*0.9, gbl_ymax*1.2)
big_xmin, _ = ax_big.get_xlim()
ax_big.set_xlim(big_xmin,1.0)

for i, ax in enumerate(axs):
    ax.set_ylim(gbl_ymin, gbl_ymax)
    
    if i != 0 and i != 3:
        ax.set_yticklabels([])
    
    else:
        ax.set_ylabel(r'$M_{\rm DM}^{\rm AC}/M_{\rm DM}^{\rm Hydro}$',fontsize=15)
    
    if i == 2 or i == 3 or i==4:
        ax.set_xlabel(r'${\rm Radius}/R_{\rm 200}$',fontsize=15)
        # ax.set_xlabel(r'${\rm Radius~[kpc]}$',fontsize=15)
    else:
        ax.set_xticklabels([])

ax_big.set_ylabel(r'$M_{\rm DM}^{\rm AC}/M_{\rm DM}^{\rm Hydro}$')
ax_big.set_xlabel(r'${\rm Radius}/R_{\rm 200}$')
# ax_big.set_xlabel(r'${\rm Radius~[kpc]}$')

plt.tight_layout()
plt.subplots_adjust(wspace=0.0,hspace=0.025)
plt.savefig(f'./figs/AC_params_weighted.pdf',bbox_inches='tight')
# plt.savefig(f'./figs/AC_params_fixed_Aw_Garcia.pdf',bbox_inches='tight')
