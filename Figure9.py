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
    
h = 0.6909

rvir = np.load('./data/rvir.npy') / h

cmap = cmr.get_sub_cmap('cmr.iceburn', 0.2, 0.8, N=1024)

fig = plt.figure(figsize=(6, 5))
ax_big = plt.gca()

x = np.load('./data/F6_median_x.npy') * np.mean(rvir)
median = np.load('./data/F6_median_y.npy')
low = np.load('./data/F6_low_y.npy'   )
high = np.load('./data/F6_high_y.npy'  )

ax_big.plot(x, median, color=cmap(0.75), lw=2.75, ls='-')
ax_big.fill_between(x, low, high, color=cmap(0.75), alpha=0.33)

ax_big.set_xscale('log')

legend_elements = [
    (cmap(0.75), cmap(0.75), '-'),
    ('white', cmap(0.25), ':'),
]

legend_labels = [
    r'${\rm DREAMS}~({\rm This~Work})$',
    r'${\rm FIRE~(Hussein\!+\!2025)}$'
]
leg = ax_big.legend(legend_elements, legend_labels, loc='upper right', frameon=False,
                    handler_map={tuple: HandlerLineWithFill()},
                    fontsize=15)

colors = [cmap(0.75), cmap(0.25)]
for index, text in enumerate(leg.get_texts()):
    text.set_color(colors[index])
ax_big.axhline(1.0, color='gray', ls='--', lw=3, alpha=0.5, zorder=-1)
    
big_xmin, _ = ax_big.get_xlim()
ax_big.set_xlim(big_xmin,10**2)
    
sims = ['f','c','i','b','m','w']
ratios = []
xs = []
j =0 
for sim in sims:
    x = np.load(f'./data/abdelaziz/rf_m12'+sim+'_corrected_fb_galpy_A_W.npy')
    xs.append(x[:2470])
    ac = np.load(f'./data/abdelaziz/AC_ratio_FIRE_m12{sim}.npy')
    ratios.append(ac[:2470])

    ax_big.plot(x, ac, color=cmap(0.25), lw=2, ls=':')
    
ratios = np.vstack(ratios)
y_mean  = np.mean(ratios, axis=0)
y_upper = np.percentile(ratios, 16, axis=0)#np.max(ratios, axis=0)
y_lower = np.percentile(ratios, 84, axis=0)#np.min(ratios, axis=0)
# ax_big.plot(xs[0], y_mean, color=cmap(0.75), lw=3, ls=':', zorder=999)
# ax_big.fill_between(xs[0], y_lower, y_upper, color=cmap(0.75), alpha=0.25, zorder=999)
# ax_big.plot(xs[0], y_lower, color=cmap(0.75), linestyle=':', zorder=999)
# ax_big.plot(xs[0], y_upper, color=cmap(0.75), linestyle=':', zorder=999)

# ax_big.fill_between(hussein_fire_x, hussein_fire_up, hussein_fire_down, color=cmap(0.75),alpha=0.25,ls='-',zorder=0)
# ax_big.plot(hussein_fire_x, hussein_fire_up, color=cmap(0.75), ls=':')
# ax_big.plot(hussein_fire_x, hussein_fire_down, color=cmap(0.75), ls=':')

ax_big.set_ylabel(r'$M_{\rm DM}^{\rm AC}/M_{\rm DM}^{\rm Hydro}$')
ax_big.set_xlabel(r'${\rm Radius~[kpc]}$')

ymin, _ = ax_big.get_ylim()
ax_big.set_ylim(0.51, 1.99)

plt.tight_layout()
plt.subplots_adjust(wspace=0.0,hspace=0.025)
plt.savefig(f'./figs/AC_params_comp_abdelaziz.pdf',bbox_inches='tight')