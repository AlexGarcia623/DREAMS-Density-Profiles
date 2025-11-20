import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import KDTree
import cmasher as cmr
from cmasher.utils import combine_cmaps
import h5py

import galaxy_transform as gt

import tutorial

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.linewidth'] = 2.25

auxtag  = 'MW_zooms'
savetag = 'DREAMS_CDM_zoom'
basedir = '/standard/DREAMS/'
sim     = 'SB5'
DM      = 'CDM'
snap_path  = basedir + 'Sims/'+DM+'/'+auxtag+'/'+sim+'/'
group_path = basedir + 'FOF_Subfind/'+DM+'/'+auxtag+'/'+sim+'/'
snapnr         = 90 ## Only works for 90
h              = 0.6909
DesNgb         = 32
scf            = tutorial.get_scf(snap_path, snapnr, 0)

sim_params = tutorial.get_params(basedir+'Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt')
sim_params = [sim_params[:,2], sim_params[:,3], sim_params[:,4], sim_params[:,0], sim_params[:,1]]

def get_boxes(params):
    box1 = np.argmin(params)
    box6 = np.argmax(params)
    
    p2 = np.percentile(params,20)
    p3 = np.percentile(params,40)
    p4 = np.percentile(params,60)
    p5 = np.percentile(params,80)
    
    box2 = np.argmin( np.abs(params - p2) )
    box3 = np.argmin( np.abs(params - p3) )
    box4 = np.argmin( np.abs(params - p4) )
    box5 = np.argmin( np.abs(params - p5) )
    
    return box1, box2, box3, box4, box5, box6

boxes = [
    *get_boxes(sim_params[0]),
    *get_boxes(sim_params[1]),
    *get_boxes(sim_params[2]),
    *get_boxes(sim_params[3]),
    *get_boxes(sim_params[4]),
]

param_name = ['sn1','sn2','agn','omega_m','sigma8']

fig, axs =plt.subplots(5,6,figsize=(10.5,8.5))
axs = axs.flatten()

white_to_black = cmr.get_sub_cmap('cmr.arctic_r', 0.0, 0.9)
colorful       = cmr.get_sub_cmap('cmr.ember', 0.1, 1.0)
combined       = combine_cmaps(*[white_to_black, colorful], nodes=[0.5])

fname = './data/pretty_dm_picture.hdf5'

for index, box in enumerate(boxes):
    print(index)
    with h5py.File(fname,'r') as f:
        this_data = np.array(f[f'{index:03d}'])
    
    ax  = axs[index]
    cax = ax.imshow(this_data, norm='log', rasterized=True, cmap=combined)
    
for ax in axs:
    # ax.axis('off')
    ax.set_yticks([])
    ax.set_xticks([])

axs[0] .text(-0.1,0.5,r'$\bar{e}_w$',ha='right',va='center',transform=axs[0].transAxes)
axs[6] .text(-0.1,0.5,r'$\kappa_w$',ha='right',va='center',transform=axs[6].transAxes)
axs[12].text(-0.1,0.5,r'$\epsilon_{f,\,{\rm high}}$',ha='right',va='center',transform=axs[12].transAxes)
axs[18].text(-0.1,0.5,r'$\Omega_{\rm M}$',ha='right',va='center',transform=axs[18].transAxes)
axs[24].text(-0.1,0.5,r'$\sigma_8$',ha='right',va='center',transform=axs[24].transAxes)

axs[0].text(0.5,1.05,r'${\rm Parameter~Minimum}$',ha='center',va='bottom',transform=axs[0].transAxes, fontsize=10)
axs[1].text(0.5,1.05,r'$20^{\rm th}~{\rm Percentile}$',ha='center',va='bottom',transform=axs[1].transAxes, fontsize=10)
axs[2].text(0.5,1.05,r'$40^{\rm th}~{\rm Percentile}$',ha='center',va='bottom',transform=axs[2].transAxes, fontsize=10)
axs[3].text(0.5,1.05,r'$60^{\rm th}~{\rm Percentile}$',ha='center',va='bottom',transform=axs[3].transAxes, fontsize=10)
axs[4].text(0.5,1.05,r'$80^{\rm th}~{\rm Percentile}$',ha='center',va='bottom',transform=axs[4].transAxes, fontsize=10)
axs[5].text(0.5,1.05,r'${\rm Parameter~Maximum}$',ha='center',va='bottom',transform=axs[5].transAxes, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.01, right=0.9)

cbar_ax = fig.add_axes([0.92, 0.1, 0.0275, 0.8])
fig.colorbar(cax, cax=cbar_ax, label=r'${\rm Density}~[M_\odot/{\rm kpc}^3]$', extend='both', extendfrac=0.025) 

plt.savefig('./figs/dm_pretty.pdf',bbox_inches='tight')
