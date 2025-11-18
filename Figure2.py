import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.colors import LogNorm
import h5py
import cmasher as cmr
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase

import matplotlib.gridspec as gridspec

from tutorial import get_params

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
mpl.rcParams['axes.linewidth'] = 2.25
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 1.0
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4.5
mpl.rcParams['ytick.minor.size'] = 4.5
mpl.rcParams['xtick.top']   = True
mpl.rcParams['ytick.right'] = True

in_file = './data/CDM_density_profiles.hdf5'

basedir    = '/standard/DREAMS/'
sim_params = get_params(basedir+'Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt')

cmaps = ['cmr.iceburn']*5
ps    = [r'$\bar{e}_w$', r'$\kappa_w$', r'$\epsilon_{f,\,{\rm high}}$', r'$\Omega_{\rm M}$', r'$\sigma_8$']
# ps    = [r'$A_{\rm SN1}$',r'$A_{\rm SN2}$',r'$A_{\rm AGN}$',r'$\Omega_{\rm M}$',r'$\sigma_8$']
label = [r'$\log A_{\rm SN2}$', r'$\log A_{\rm SN2}$',r'$\log A_{\rm AGN}$',r'$\log\Omega_M$',r'$\log\sigma_8$']

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

weights = np.load('./data/arya_weights_binned.npy')
params = sim_params[:,2:5]
ws = np.zeros(1024)
for i in range(1024):
    idx   = get_bin_idx(params[i]) 
    ws[i] = weights[idx[0]][idx[1]][idx[2]]

# ws /= np.max(ws)

plt.figure(figsize=(6,3))

plt.hist(ws, bins=30)
plt.yscale('log')

two_sigma_val = np.percentile(ws,25)

plt.axvline(two_sigma_val, color='k', ls='--')

plt.xlabel(r'${\rm Weight~Value}$')
plt.ylabel(r'${\rm Count}$')

plt.tight_layout()
plt.savefig('./figs/weights.pdf', bbox_inches='tight')

plt.close()

rvir = np.load('./data/rvir.npy') / 0.6909

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
    
def fill_plot(ax,x,avgs,col,tag=''):
    scatter_d = np.percentile(avgs, 25, axis=0)
    scatter_u = np.percentile(avgs, 75, axis=0)
    
    median = np.mean(avgs, axis=0)
    
    at_1pct = 0#np.argmin(np.abs(x - 0.01))
    
    med=median[at_1pct]
    upp=scatter_u[at_1pct]
    low=scatter_d[at_1pct]
    
    ss='{'
    se='}'
    
    print(f"\t{tag}")
    print(f"\t\tMedian at {x[at_1pct]:0.4f} Rvir = {med:0.3f}_{ss}-{med-low:0.3f}{se}^{ss}{upp-med:0.3f}{se}")
    
    ax.fill_between(x, scatter_d, scatter_u, color=col, alpha=0.33)
    
def weighted_std(values, weights):
    average  = np.average(values, weights=weights, axis=0)
    return np.sqrt(np.average((values-average)**2, weights=weights, axis=0))
    
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
    
sim_params = [sim_params[:,2], sim_params[:,3], sim_params[:,4], sim_params[:,0], sim_params[:,1]]
    
names     = ['sn1', 'sn2', 'agn', 'omega_m', 'sigma_8', 'halo_mass']
    
for which in range(0,5):
    print(names[which])
    ax = axs[which]
    
    xstart = 2.1
    ystart = 10**(3.15)
    xlen   = xstart+15
    ylen   = 10**(np.log10(ystart)+2.75)
    
    param = sim_params[which]
    if which in [0, 1, 2]:
        param = np.log10(param)
    dr         = 10 ## percent
    min_param  = np.percentile(param,dr)
    max_param  = np.percentile(param,100-dr)
    avg1_param = np.percentile(param,50-dr/2)
    avg2_param = np.percentile(param,50+dr/2)    
    norm_param = (param - np.min(param)) / (np.max(param) - np.min(param))
    cmap = cmr.get_sub_cmap(cmaps[which], 0.2, 0.8, N=1024)

    colors = [cmap(value) for value in norm_param]

    lower_avg  = []
    upper_avg  = []
    middle_avg = []
    all_profs  = []
    
    lower_rvir  = []
    upper_rvir  = []
    middle_rvir = []
    
    dens_1kpc   = np.zeros(1024)
    dens_10kpc  = np.zeros(1024)
    dens_100kpc = np.zeros(1024)
    
    with h5py.File(in_file, 'r') as file:
        for index, box in enumerate(np.arange(0,1024)):
            this_param = param[index]
            lower = this_param <= min_param
            upper = this_param >= max_param
            middle = (this_param >= avg1_param) & (this_param <= avg2_param)
            plot = lower or upper or middle
            
            try:
                this_box = file[f'box_{box:04d}']
                density = np.array(this_box['density'])
                radius  = np.array(this_box['radius']) #/ rvir[box]
            except:
                print(f'box_{box:04d} did not work')
                dens_1kpc[index]   = np.nan
                dens_10kpc[index]  = np.nan
                dens_100kpc[index] = np.nan
                continue

            h       = 0.6909
            rmin    = 0.305 / h * 2.8
            rmax    = 600
            # if rvir[box] > rmax:
            #     print(box, rvir[box])
            
            density = density[(radius > rmin) & (radius < rmax)]
            radius  = radius [(radius > rmin) & (radius < rmax)]
            
            if which == 0:
                hydro_prof = interp1d(radius/rvir[box], density , fill_value='extrapolate')
                x = 10**np.linspace(-2.3, np.log10(3), 200)
                all_profs.append(hydro_prof(x))

            dens_1kpc[index]   = density[np.argmin(np.abs(radius/rvir[box] - 0.01))]
            dens_10kpc[index]  = density[np.argmin(np.abs(radius/rvir[box] - 0.1))]
            dens_100kpc[index] = density[np.argmin(np.abs(radius/rvir[box] - 1.0))]
            if plot:
                if lower:
                    col = cmap(0)
                    lower_avg.append(density)
                    lower_rvir.append(rvir[box])
                elif upper: 
                    col = cmap(0.99)
                    upper_avg.append(density)
                    upper_rvir.append(rvir[box])
                else: 
                    col = cmap(0.5)
                    middle_avg.append(density)
                    middle_rvir.append(rvir[box])

            if which == 0:
                if BW:
                    ax_big.plot(radius/rvir[box], density, color='w', lw=1, rasterized=True, alpha=0.01)
                else:
                    ax_big.plot(radius/rvir[box], density, color='k', lw=1, rasterized=True, alpha=0.01)

    lower_avg  = np.array(lower_avg )
    upper_avg  = np.array(upper_avg )
    middle_avg = np.array(middle_avg)
    
    lower_rvir  = np.array(lower_rvir )
    upper_rvir  = np.array(upper_rvir )
    middle_rvir = np.array(middle_rvir)
    
    lower_avg_copy  = lower_avg 
    upper_avg_copy  = upper_avg 
    middle_avg_copy = middle_avg

    lower_avg  = np.mean(lower_avg , axis=0)
    upper_avg  = np.mean(upper_avg , axis=0)
    middle_avg = np.mean(middle_avg, axis=0)

    lower_rvir  = np.mean(lower_rvir )
    upper_rvir  = np.mean(upper_rvir )
    middle_rvir = np.mean(middle_rvir)
    
    ax.plot(radius / lower_rvir, lower_avg /middle_avg, color=cmap(0.00), lw=2.75, ls='--')
    ax.plot(radius / upper_rvir, upper_avg /middle_avg, color=cmap(0.99), lw=2.75, ls=':')
    ax.axhline(1.0, color='gray', ls='-', alpha=0.75)
    
    fill_plot(ax, radius/ lower_rvir, lower_avg_copy /middle_avg, cmap(0.00), tag='lower')
    fill_plot(ax, radius/ upper_rvir, upper_avg_copy /middle_avg, cmap(0.99), tag='upper')
        
    ax.set_xscale('log')
    
    xmin, _ = ax.get_xlim()
    ax.set_xlim(xmin,10**(-0.9))
    ax.set_ylim(0.1,2.55)
    
    if ax in [ax3, ax4]:
        ax.set_xticklabels([])
    
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
                
        median  = np.average(all_profs, weights=ws, axis=0)
        low     = median-weighted_std(all_profs, weights=ws)
        high    = median+weighted_std(all_profs, weights=ws)
        scatter = np.log10(high) - np.log10(low)
        ax_big.plot(x, median, color=cmap(0.75), lw=2, alpha=0.75)
        ax_big.fill_between(x, low, high, color=cmap(0.75), alpha=0.25)
        ax_big.plot(x, low , color=cmap(0.75), lw=0.5, alpha=0.25)
        ax_big.plot(x, high, color=cmap(0.75), lw=0.5, alpha=0.25)
        
        np.save('../CDM_DMO/data/hydro_x.npy',x)
        np.save('../CDM_DMO/data/hydro_y.npy',median)
        
        np.save('../CDM_DMO/data/hydro_yerr_low.npy' ,low)
        np.save('../CDM_DMO/data/hydro_yerr_high.npy',high)
        
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
        ax_big.set_ylim(10**(3.5),10**(9.1))
        xmin, xmax = ax_big.get_xlim()
        ax_big.set_xlim(xmin, 2)
        
        legend_elements = [
            Line2D([0], [0], color='k', lw=2, alpha=0.25, ls='-', label=r'${\rm Idv.~ Halos}$'),
            (cmap(0.75), cmap(0.75)),
        ]

        legend_labels = [
            r'${\rm Idv.~ Halos}$',
            r'$\hat{\mu} + \hat{\sigma}$'
        ]
        leg = ax_big.legend(legend_elements, legend_labels, loc='upper right', frameon=False,
                            fontsize=12,
                            handler_map={tuple: HandlerLineWithFill()})
        
        colors = ['gray', cmap(0.75)]
        for index, text in enumerate(leg.get_texts()):
            text.set_color(colors[index])
        
    ax.text(0.95,0.925,ps[which], color='w' if BW else 'k',
            transform=ax.transAxes, ha='right', va='top')

    print('')

gbl_ymin = np.inf
gbl_ymax = -np.inf
for ax in axs:
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    
    if ymin < gbl_ymin:
        gbl_ymin = ymin
    if ymax > gbl_ymax:
        gbl_ymax = ymax
        
for i, ax in enumerate(axs):
    ax.set_ylim(gbl_ymin, gbl_ymax)
    
    if i != 0 and i != 3:
        ax.set_yticklabels([])
    
    else:
        ax.set_ylabel(r'$\rho_{\rm var}/\rho_{\rm avg}$')
    
    if i == 2 or i == 3 or i==4:
        ax.set_xlabel(r'${\rm Radius}/R_{\rm 200}$')
    
# ax_big.text(0.95, 0.95, r'${\rm All~Halos}$', color='w' if BW else 'k',
#              transform=ax_big.transAxes, ha='right', va='top')
# ax_big.text(0.95, 0.875, r'${\rm Median}$', color='green',
#              transform=ax_big.transAxes, ha='right', va='top')
    
ax_big.set_xlabel(r'${\rm Radius/}R_{\rm 200}$')
ax_big.set_ylabel(r'${\rm Density~}[M_\odot/{\rm kpc}^3]$')

plt.tight_layout()
plt.subplots_adjust(wspace=0.0,hspace=0.025)

tag = 'density'
plt.savefig(f'./figs/Figure2.pdf',bbox_inches='tight')
