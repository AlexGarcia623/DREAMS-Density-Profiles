import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import cmasher as cmr
from cmasher.utils import combine_cmaps
import h5py

import galaxy_transform as gt

import tutorial

## Torreylabtools propriety
import visualization.contour_makepic as makepic
import util.calc_hsml as calc_hsml

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# plt.rcParams['axes.facecolor'] = 'black'
# plt.rcParams['figure.facecolor'] = 'black'

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

fig, axs =plt.subplots(6,6,figsize=(16,10))
axs = axs.flatten()

white_to_black = cmr.get_sub_cmap('Greys', 0.0, 1.0)
colorful       = cmr.get_sub_cmap('cmr.ember', 0.0, 1.0)
combined       = combine_cmaps(white_to_black, colorful)

fname = './data/pretty_dm_picture.hdf5'

with h5py.File(fname, 'w') as f:
    pass

for index, box in enumerate(boxes):
    _s_ = f'Starting box {box}'
    print('')
    print('#'*(len(_s_) + 8))
    print('### ' + _s_ + ' ###')
    print('#'*(len(_s_) + 8))
    print('')
    ## load all data
    path = f'{snap_path}/box_{box}/snap_{snapnr:03}.hdf5'
    prt_cat = tutorial.load_particle_data(path, ['Masses', 'Coordinates', 'Velocities'], [1])

    path = f'{group_path}/box_{box}/fof_subhalo_tab_{snapnr:03}.hdf5'
    keys = ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'SubhaloPos',
            'GroupMassType', 'GroupPos', 'SubhaloLenType', 'SubhaloGrNr',
            'GroupVel','Group_R_Crit200']
    grp_cat = tutorial.load_group_data(path, keys)

    ## get only MW
    mw_idx = tutorial.get_MW_idx(grp_cat)
    _, fof_cat = tutorial.get_galaxy_data(prt_cat, grp_cat, mw_idx)

    ## load dm
    dm_mass      = prt_cat[f'PartType1/Masses'] * 1.00E+10 / h
    dm_coords    = prt_cat[f'PartType1/Coordinates'] / h
    dm_vel       = prt_cat[f'PartType1/Velocities'] * np.sqrt(scf)
    
    ## Get center
    gal_pos      = fof_cat['GroupPos'] / h
    gal_vel      = fof_cat['GroupVel'] * np.sqrt(scf)

    ## center
    dm_coords   -= gal_pos
    dm_vel      -= gal_vel

    dm_pos = dm_coords
    dm_rad = np.sqrt( dm_pos[:,0]**2 + dm_pos[:,1]**2 + dm_pos[:,2]**2 )
    
    rvir = fof_cat['Group_R_Crit200']/h
    
    within_2rvir = dm_rad < (2.5*rvir)

    dm_hsml = calc_hsml.get_particle_hsml( dm_pos[within_2rvir,0], dm_pos[within_2rvir,1], dm_pos[within_2rvir,2], DesNgb=32  )

    size      = rvir*1.5
    n_pixels  = 4000
    max_den   = 10**(8.5)
    dynrng    = 10**(3)
    
    massmap,image = makepic.contour_makepic( dm_pos[within_2rvir,0], dm_pos[within_2rvir,1], dm_pos[within_2rvir,2],
        dm_hsml, dm_mass[within_2rvir] ,
        xlen = size,
        pixels = n_pixels,
        set_aspect_ratio = 1,
        set_maxden = max_den, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
        set_dynrng = dynrng
    )
    
    with h5py.File(fname,'r+') as f:
        f.create_dataset(f'{index:03d}', data=massmap)
    
    ax  = axs[index]
    cax = ax.imshow(massmap, norm='log', rasterized=True, cmap=combined)
        
    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.2)
    plt.savefig('./figs/dm_pretty_test.pdf',bbox_inches='tight')
