import sys, os
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy.spatial import KDTree
from tqdm import tqdm

import tutorial

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

auxtag  = 'MW_zooms'
savetag = 'DREAMS_CDM_zoom'
basedir = '/standard/DREAMS/'
sim     = 'SB5'
DM      = 'CDM'
snap_path  = basedir + 'Sims/'+DM+'/'+auxtag+'/'+sim+'/'
group_path = basedir + 'FOF_Subfind/'+DM+'/'+auxtag+'/'+sim+'/'
snapnr         = 90 ## Only works for 90
h              = 0.6909
rmin           = 0.305 / h * 2.8
rmax           = 600
sphere_samples = int(1e3)
radial_samples = 300
DesNgb         = 32
scf            = tutorial.get_scf(snap_path, snapnr, 0)

sim_params = tutorial.get_params(basedir+'Parameters/CDM/'+auxtag+'/CDM_TNG_MW_SB5.txt')

out_file_name = f'./data/density_profiles_{rank}.hdf5'

with h5py.File(out_file_name, 'w') as f:
    1==1
    
def process_box(box):
    print(f'Rank {rank} starting box {box}')
    
    path = f'{snap_path}/box_{box}/snap_{snapnr:03}.hdf5'
    all_part_cat = tutorial.load_particle_data(path, ['Masses', 'Coordinates'], [1,2])
    
    path = f'{group_path}/box_{box}/fof_subhalo_tab_{snapnr:03}.hdf5'
    keys = ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'SubhaloPos',
            'GroupMassType', 'GroupPos', 'SubhaloLenType', 'SubhaloGrNr',
            'Group_R_Crit200']
    grp_cat = tutorial.load_group_data(path, keys)
    
    mw_idx = tutorial.get_MW_idx(grp_cat)
    _, fof_cat = tutorial.get_galaxy_data(all_part_cat, grp_cat, mw_idx)
    
    dm1_masses  = all_part_cat[f'PartType1/Masses'] * 1.00E+10 / h
    dm2_masses  = all_part_cat[f'PartType2/Masses'] * 1.00E+10 / h
    dm1_coords  = all_part_cat[f'PartType1/Coordinates'] * scf / h
    dm2_coords  = all_part_cat[f'PartType2/Coordinates'] * scf / h
    
    all_dm_masses = np.concatenate([dm1_masses, dm2_masses])
    all_dm_coords = np.concatenate([dm1_coords, dm2_coords])
    
    gal_pos = fof_cat['GroupPos'] * scf / h
    all_dm_coords -= gal_pos
    
    all_dm_rads = np.sqrt(all_dm_coords[:,0]**2 + all_dm_coords[:,1]**2 + all_dm_coords[:,2]**2)
    
    rvir = fof_cat['Group_R_Crit200'] * scf / h
    
    keep_within_4rvir = all_dm_rads < rmax*1.5
        
    all_dm_coords = all_dm_coords[keep_within_4rvir, :]
    
    all_r = np.logspace(np.log10(rmin), np.log10(rmax), radial_samples)
    all_density = np.zeros(len(all_r))
    all_spread  = np.zeros(len(all_r))
    
    tree = KDTree(all_dm_coords)
    for index, r in tqdm(enumerate(all_r)):
        points  = tutorial.fibonacci_sphere(sphere_samples, r)
        density, spread = tutorial.calc_density(tree, all_dm_masses, points, DesNgb, spread=True)
        all_spread[index]  = spread
        all_density[index] = density
        
    with h5py.File(out_file_name, 'r+') as f:
        this_box = f.create_group(f'box_{box:04d}')
        this_box.create_dataset('radius' , data=all_r)
        this_box.create_dataset('density', data=all_density)
        this_box.create_dataset('spread' , data=all_spread)
        
boxes = np.arange(0, 1024)
boxes_per_rank = np.array_split(boxes, size)

for box in boxes_per_rank[rank]:
    process_box(box)