import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import h5py
import cmasher as cmr
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase

import matplotlib.gridspec as gridspec

from tutorial import get_params, get_box_size

from scipy.interpolate import interp1d

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
mpl.rcParams['axes.linewidth'] = 2.25 * 0.75
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5* 0.75
mpl.rcParams['ytick.major.width'] = 1.5* 0.75
mpl.rcParams['xtick.minor.width'] = 1.0* 0.75
mpl.rcParams['ytick.minor.width'] = 1.0* 0.75
mpl.rcParams['xtick.major.size'] = 8*0.75
mpl.rcParams['ytick.major.size'] = 8*0.75
mpl.rcParams['xtick.minor.size'] = 4.5*0.75
mpl.rcParams['ytick.minor.size'] = 4.5*0.75
mpl.rcParams['xtick.top']   = True
mpl.rcParams['ytick.right'] = True

in_file  = '../CDM/CDM_density_profiles.hdf5'
in2_file = 'CDM_DMO_density_profiles.hdf5'

basedir    = '/standard/DREAMS/'
sim_params = get_params(basedir+'Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt')

auxtag  = 'MW_zooms'
savetag = 'DREAMS_CDM_zoom'
basedir = '/standard/DREAMS/'
sim     = 'SB5_Nbody'
DM      = 'CDM'
snap_path  = basedir + 'Sims/'+DM+'/'+auxtag+'/'+sim+'/'
group_path = basedir + 'FOF_Subfind/'+DM+'/'+auxtag+'/'+sim+'/'
snapnr = 90

cmaps = ['cmr.iceburn']*5
# ps    = [r'$A_{\rm SN1}$',r'$A_{\rm SN2}$',r'$A_{\rm AGN}$',r'$\Omega_{\rm M}$',r'$\sigma_8$']
ps    = [r'$\bar{e}_w$', r'$\kappa_w$', r'$\epsilon_{f,\,{\rm high}}$', r'$\Omega_{\rm M}$', r'$\sigma_8$']
label = [r'$\log A_{\rm SN}$', r'$\log A_{\rm SN2}$',r'$\log A_{\rm AGN}$',r'$\log\Omega_M$',r'$\log\sigma_8$']


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

axs = [ax3, ax4, ax5, ax6, ax7, ax8]

h = 0.6909

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

weights = np.load('../CDM/data/arya_weights_binned.npy')
params = sim_params[:,2:5]
ws = np.zeros(1024)
for i in range(1024):
    idx   = get_bin_idx(params[i]) 
    ws[i] = weights[idx[0]][idx[1]][idx[2]]

remove_idx = [796, 10,15,448,476,503,656,704, 410,547,581,781,808,829,984]

ws_filtered = np.delete(ws, remove_idx)

two_sigma_val = np.percentile(ws_filtered,25)
    
def fill_plot(ax,x,avgs,col):
    scatter_d = np.percentile(avgs, 16, axis=0)
    scatter_u = np.percentile(avgs, 84, axis=0)
    
    median = np.median(avgs, axis=0)
    
    ax.fill_between(x, scatter_d, scatter_u, color=col, alpha=0.33)
    ax.plot(x, scatter_d, color=col, alpha=0.33, lw=1)
    ax.plot(x, scatter_u, color=col, alpha=0.33, lw=1)
    
def weighted_std(values, weights):
    average  = np.average(values, weights=weights, axis=0)
    return np.sqrt(np.average((values-average)**2, weights=weights, axis=0))

class HandlerLineWithFill(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        fill_color, line_color, ls = orig_handle
        # Create a rectangle (fill)
        fill = Rectangle([xdescent, ydescent], width, height,
                         facecolor=fill_color, edgecolor='none', alpha=0.3, transform=trans)
        # Create a line over the rectangle
        line = Line2D([xdescent, xdescent + width],
                      [ydescent + height / 2, ydescent + height / 2],
                      color=line_color, lw=2, transform=trans, ls=ls)
        return [fill, line]
    
rvir_dmo   = np.load('./data/rvir.npy')
rvir_hydro = np.load('../CDM/data/rvir.npy') / h

Omega_m = sim_params[:,0]
Omega_b = 0.046
sim_params = [sim_params[:,2], sim_params[:,3], sim_params[:,4], sim_params[:,0], sim_params[:,1]]
for which in range(0,5):
    ax = axs[which]
    param = sim_params[which]
    param = np.log10(param)
    dr         = 10 ## percent
    min_param  = np.percentile(param,dr)
    max_param  = np.percentile(param,100-dr)
    avg1_param = np.percentile(param,50-dr/2)
    avg2_param = np.percentile(param,50+dr/2)    
    norm_param = (param - np.min(param)) / (np.max(param) - np.min(param))
    cmap = cmr.get_sub_cmap(cmaps[which], 0.2, 0.8, N=1024)

    omega_m = sim_params[3]
    
    colors = [cmap(value) for value in norm_param]

    lower_avg  = []
    upper_avg  = []
    middle_avg = []
    
    all_profs = []
    
    with h5py.File(in_file, 'r') as file:
        with h5py.File(in2_file, 'r') as file2:
            for index, box in enumerate(np.arange(0,1024)):
                if box in [796]: ## DNE
                    continue
                if box in [10,15,448,476,503,656,704]: ## select wrong subhalo
                    continue
                if box in [410,547,581,781,808,829,984]: ## select wrong halo
                    continue
                this_param = param[index]
                lower = this_param <= min_param
                upper = this_param >= max_param
                middle = (this_param >= avg1_param) & (this_param <= avg2_param)
                plot = lower or upper or middle
                
                this_box  = file[f'box_{box:04d}']
                this_box2 = file2[f'box_{box:04d}']
                
                ## 1 -> hydro 
                ## 2 -> dmo 
                dmo_rvir  = rvir_dmo[box] if box != 581 else 151.63839596676434
                radius    = np.array(this_box['radius'])  / rvir_hydro[box]
                radius2   = np.array(this_box2['radius']) / rvir_dmo[box]
                density   = np.array(this_box['density'])
                density2  = np.array(this_box2['density']) # * ((Omega_m[index]-Omega_b) / Omega_m[index])
                
                if density2[0] < 1e7:
                    print(box)
                
                density  = density 
                density2 = density2
                radius   = radius  
                radius2  = radius2 
                
                hydro_prof = interp1d(radius , density , fill_value='extrapolate')
                dmo_prof   = interp1d(radius2, density2, fill_value='extrapolate')
                
                x = 10**np.linspace(-2.3, np.log10(3), 100)

                ratio = hydro_prof(x) / dmo_prof(x)
                
                all_profs.append(dmo_prof(x))
                
                if which == 0:
                    ax_big.plot(radius2, density2, color='w' if BW else 'k', lw=1, alpha=0.01, rasterized=True)
                    ax_big.set_xscale('log')
                    ax_big.set_yscale('log')
                    
                if plot:
                    if lower:
                        col = cmap(0.00)
                        lower_avg.append(ratio)
                    elif upper: 
                        col = cmap(0.99)
                        upper_avg.append(ratio)
                    else: 
                        col = cmap(0.5)
                        middle_avg.append(ratio)
                    
    lower_avg  = np.array(lower_avg )
    upper_avg  = np.array(upper_avg )
    middle_avg = np.array(middle_avg)
    
    all_profs = np.array(all_profs)
    
    lower_avg_copy  = lower_avg 
    upper_avg_copy  = upper_avg 
    middle_avg_copy = middle_avg

    lower_avg  = np.nanmedian(lower_avg , axis=0)
    upper_avg  = np.nanmedian(upper_avg , axis=0)
    middle_avg = np.nanmedian(middle_avg, axis=0)
    
    # ax.plot(radius, lower_avg, color=cmap(0.00), lw=3, rasterized=True, alpha=1, ls='--')
    # ax.plot(radius, upper_avg, color=cmap(0.99), lw=3, rasterized=True, alpha=1, ls=':')
    
    ax.plot(x, lower_avg, color=cmap(0.00), lw=3, rasterized=True, alpha=1, ls='--')
    ax.plot(x, upper_avg, color=cmap(0.99), lw=3, rasterized=True, alpha=1, ls=':')

    # fill_plot(ax, radius, lower_avg_copy , cmap(0.00))
    # fill_plot(ax, radius, upper_avg_copy , cmap(0.99))

    fill_plot(ax, x, lower_avg_copy , cmap(0.00))
    fill_plot(ax, x, upper_avg_copy , cmap(0.99))
    
    ax.set_xscale('log')
    xmin, _ = ax.get_xlim()
    ax.set_xlim(xmin,10**(-0.7))
    
    ax.axhline(1.0, color='gray', alpha=0.75)
    
    ax.text(0.95,0.925,ps[which], color='w' if BW else 'k',
            transform=ax.transAxes, ha='right', va='top')
    if which == 0:
        ax8.plot([0,0],[1,1], alpha=1.0, label=r'${\rm Upper~%s}$'%dr + r'$\%$', ls=':',
                 color=cmap(0.99), lw=2.75)
        ax8.plot([0,0],[1,1], alpha=1.0, label=r'${\rm Lower~%s}$'%dr + r'$\%$', ls='--',
                 color=cmap(0.01), lw=2.75)
        
        leg = ax8.legend(frameon=False, loc='center', fontsize=14,
                         labelspacing=0.05,handletextpad=0.1)
        colors = [cmap(0.99), cmap(0.01)]
        for index, text in enumerate(leg.get_texts()):
            text.set_color(colors[index])
            
        median  = np.average(all_profs, weights=ws_filtered, axis=0)
        low     = median-weighted_std(all_profs, weights=ws_filtered)
        high    = median+weighted_std(all_profs, weights=ws_filtered)
        scatter = np.log10(high) - np.log10(low)
        ax_big.plot(x, median, color='green', lw=2, alpha=0.75)
        ax_big.fill_between(x, low, high, color='green', alpha=0.25)
        ax_big.plot(x, low , color='green', lw=0.25)
        ax_big.plot(x, high, color='green', lw=0.25)
        
        dens_1kpc   = scatter[np.argmin(np.abs(x - 0.01))]
        dens_10kpc  = scatter[np.argmin(np.abs(x - 0.1))]
        dens_100kpc = scatter[np.argmin(np.abs(x - 1))]
        
        ax_big.text(0.05,0.22-0.05,r'$\hat{\sigma}_{0.01 R_{\rm 200}}$',
                     transform=ax_big.transAxes, fontsize=13)
        ax_big.text(0.25,0.22-0.05,r'$=%0.2f~{\rm dex}$' %dens_1kpc,
                     transform=ax_big.transAxes, fontsize=13)
        ax_big.text(0.05,0.16-0.04,r'$\hat{\sigma}_{0.1 R_{\rm 200}}$',
                     transform=ax_big.transAxes, fontsize=13)
        ax_big.text(0.25,0.16-0.04,r'$=%0.2f~{\rm dex}$' %dens_10kpc,
                     transform=ax_big.transAxes, fontsize=13)
        ax_big.text(0.05,0.10-0.03,r'$\hat{\sigma}_{R_{\rm 200}}$',
                     transform=ax_big.transAxes, fontsize=13)
        ax_big.text(0.25,0.10-0.03,r'$=%0.2f~{\rm dex}$' %dens_100kpc,
                     transform=ax_big.transAxes, fontsize=13)
        
        ax_big.set_yscale('log')
        ax_big.set_xscale('log')
        ax_big.set_ylim(10**(3.5),10**(8.8))
        xmin, xmax = ax_big.get_xlim()
        ax_big.set_xlim(xmin, 2)
        
        hydro_x = np.load('./data/hydro_x.npy')
        hydro_y = np.load('./data/hydro_y.npy')
        hydro_y_low   = np.load('./data/hydor_yerr_low.npy')
        hydro_y_upper = np.load('./data/hydor_yerr_high.npy')
        
        ax_big.plot(hydro_x, hydro_y, color='red' if BW else 'mediumorchid', alpha=0.5, ls='--', lw=3)
        ax_big.fill_between(hydro_x, hydro_y_low,hydro_y_upper, color='red' if BW else 'mediumorchid', alpha=0.2)
        
        legend_elements = [
            Line2D([0], [0], color='white' if BW else 'k', lw=2, alpha=0.25, ls='-', label=r'${\rm Idv.~ Halos}$'),
            ('green', 'green', '-'),
            ('mediumorchid','mediumorchid','--')
        ]

        legend_labels = [
            r'${\rm Idv.~ Halos~DMO}$',
            r'${\rm DMO}~\hat{\mu}+\hat{\sigma}$',
            r'${\rm Hydro~\hat{\mu}+\hat{\sigma}}$'
        ]
        leg = ax_big.legend(legend_elements, legend_labels, loc='upper right', frameon=False,
                            fontsize=11,
                            handler_map={tuple: HandlerLineWithFill()})
        
        colors = ['gray', 'green', 'red' if BW else 'mediumorchid']
        for index, text in enumerate(leg.get_texts()):
            text.set_color(colors[index])
        
gbl_ymin = []
gbl_ymax = []
    
for i, ax in enumerate(axs[:-1]):
    gbl_ymin.append(ax.get_ylim()[0]); gbl_ymax.append(ax.get_ylim()[1])
        
for i, ax in enumerate(axs[:-1]):
    
    ax.set_ylim(
        np.min(gbl_ymin),
        np.max(gbl_ymax)
    )
    
    if i == 0 or i == 1:
        ax.set_xticklabels([])
        
    if i not in [0,3]:
        ax.set_yticklabels([])
    
    
# ax_big.text(0.95, 0.95, r'${\rm All~Halos}$', color='w' if BW else 'k',
#              transform=ax_big.transAxes, ha='right', va='top')
# ax_big.text(0.95, 0.885, r'${\rm Dark~Matter~Only}$', fontsize=12, color='w' if BW else 'k',
#              transform=ax_big.transAxes, ha='right', va='top')
    
for ax in [axs[3],axs[4],axs[2]]:
    ax.set_xlabel(r'${\rm Radius}/R_{\rm 200}$', fontsize=18)
for ax in [axs[0], axs[3]]:
    ax.set_ylabel(r'$\rho_{\rm hydro}/\rho_{\rm DMO}$', fontsize=18)
    
ax_big.set_xlabel(r'${\rm Radius}/R_{\rm 200}$')
ax_big.set_ylabel(r'${\rm Density~}[M_\odot/{\rm kpc}^3]$')

plt.tight_layout()
plt.subplots_adjust(wspace=0.0,hspace=0.025)

plt.savefig(f'./figs/comp_baryon_ratio_weighted.pdf',bbox_inches='tight')
