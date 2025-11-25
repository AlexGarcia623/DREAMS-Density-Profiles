import sys, os
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import emcee
import corner

import tutorial 

from scipy.interpolate import interp1d
from scipy.optimize import minimize, fixed_point, curve_fit, differential_evolution, least_squares

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
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
snapnr     = 90
h          = 0.6909
rmin       = 0.305 / h * 2.8

sim_params = tutorial.get_params(basedir+'Parameters/CDM/'+auxtag+'/CDM_TNG_MW_SB5.txt')

out_file_name = f'./data/CDM_baryon_contract_{rank}.hdf5'

with h5py.File(out_file_name, 'w') as f:
    1==1
    
with h5py.File(f'{snap_path}/box_0/snap_{snapnr:03d}.hdf5', 'r') as hdr_file:
    header = hdr_file['Header']
    Omega_b = header.attrs['OmegaBaryon']

def center_and_rad(pos, center):
    pos -= center
    return np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    
def baryons(box):
    print(f'Rank {rank} starting box {box}')
    # print('\tBaryons')
    
    path = f'{snap_path}/box_{box}/snap_{snapnr:03}.hdf5'
    part_cat = tutorial.load_particle_data(path, ['Masses', 'Coordinates'], [0,1,2,5])
    star_cat = tutorial.load_particle_data(path, ['Masses', 'Coordinates', 'GFM_StellarFormationTime'], [4])
    
    path = f'{group_path}/box_{box}/fof_subhalo_tab_{snapnr:03}.hdf5'
    keys = ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'SubhaloPos', 'GroupMass',
            'GroupMassType', 'GroupPos', 'SubhaloLenType', 'SubhaloGrNr', 'Group_R_Crit200',
            'Group_M_Crit200']
    grp_cat = tutorial.load_group_data(path, keys)
    
    mw_idx = tutorial.get_MW_idx(grp_cat)
    _, fof_cat = tutorial.get_galaxy_data(part_cat, grp_cat, mw_idx)
    # _, _       = tutorial.get_galaxy_data(star_cat, grp_cat, mw_idx)
    
    star_age = star_cat['PartType4/GFM_StellarFormationTime']
    
    gas_mass    = part_cat[f'PartType0/Masses'] * 1.00E+10 / h
    gas_coords  = part_cat[f'PartType0/Coordinates'] / h
    dm_mass     = part_cat[f'PartType1/Masses'] * 1.00E+10 / h
    dm_coords   = part_cat[f'PartType1/Coordinates'] / h
    dm2_mass    = part_cat[f'PartType2/Masses'] * 1.00E+10 / h
    dm2_coords  = part_cat[f'PartType2/Coordinates'] / h
    star_mass   = star_cat[f'PartType4/Masses'][star_age > 0] * 1.00E+10 / h
    star_coords = star_cat[f'PartType4/Coordinates'][star_age > 0] / h
    bh_mass     = part_cat[f'PartType5/Masses'] * 1.00E+10 / h
    bh_coords   = part_cat[f'PartType5/Coordinates'] / h
    
    gal_pos = fof_cat['GroupPos'] / h
    r200    = fof_cat['Group_R_Crit200'] / h
    
    gas_rad  = center_and_rad(gas_coords , gal_pos)
    dm_rad   = center_and_rad(dm_coords  , gal_pos)
    dm2_rad  = center_and_rad(dm2_coords , gal_pos)
    star_rad = center_and_rad(star_coords, gal_pos)
    bh_rad   = center_and_rad(bh_coords  , gal_pos)
    
    dr   = 0.5
    rmax = 300
    if rmax < r200:
        rmax = r200
    
    rs   = 10**np.linspace(np.log10(rmin),np.log10(rmax+dr),50)
    
    cum_mass_dm_only    = np.zeros(len(rs))
    cum_mass_baryons_only = np.zeros(len(rs))
    for index, r in enumerate(rs):
        gas_within_dr  = (gas_rad <= r) 
        DM_within_dr   = (dm_rad <= r)  
        DM2_within_dr  = (dm2_rad <= r) 
        star_within_dr = (star_rad <= r)
        bh_within_dr   = (bh_rad <= r)
        
        cum_mass_dm_only[index]    = np.sum([
            np.sum(dm_mass[DM_within_dr]),np.sum(dm2_mass[DM2_within_dr])
        ])
        cum_mass_baryons_only[index] = np.sum([
            np.sum(star_mass[star_within_dr]),
            np.sum(gas_mass[gas_within_dr]),
            np.sum(bh_mass[bh_within_dr])
        ])
    
    return np.log10(cum_mass_dm_only), np.log10(cum_mass_baryons_only), r200, rs
    
def dmo(box, rs):
    part_type = 1
    
    sim     = 'SB5_Nbody'
    snap_path  = basedir + 'Sims/CDM/'+auxtag+'/'+sim+'/'
    group_path = basedir + 'FOF_Subfind/CDM/'+auxtag+'/'+sim+'/'
    
    path = f'{snap_path}/box_{box}/snap_{snapnr:03}.hdf5'
    part_cat = tutorial.load_particle_data(path, ['Masses', 'Coordinates'], part_type)
    
    path = f'{group_path}/box_{box}/fof_subhalo_tab_{snapnr:03}.hdf5'
    keys = ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'GroupMassType', 
            'GroupPos', 'SubhaloLenType', 'SubhaloGrNr', 'GroupMass']
    grp_cat = tutorial.load_group_data(path, keys)
    
    mw_idx = tutorial.get_MW_idx(grp_cat)
    prt_cat, fof_cat = tutorial.get_galaxy_data(part_cat, grp_cat, mw_idx)
    
    masses  = part_cat[f'PartType{part_type}/Masses'] * 1.00E+10 / h
    coords  = part_cat[f'PartType{part_type}/Coordinates'] / h
    gal_pos = fof_cat['GroupPos'] / h
    
    dm_rad  = center_and_rad(coords, gal_pos)
    
    cum_mass_dm_only   = np.zeros(len(rs))
    for index, r in enumerate(rs):
        DM_within_dr   = (dm_rad <= r)  
        cum_mass_dm_only[index] = np.sum(masses[DM_within_dr])
    return np.log10(cum_mass_dm_only)
    
boxes = np.arange(0, 1024)
boxes_per_rank = np.array_split(boxes, size)

As = []
ws = []

for box in boxes_per_rank[rank]:
    if box == 796: ## Doesn't have DMO data
        continue
    ## Get profiles from sims
    M_enc_dm, M_enc_stars, r200, rs = baryons(box)
    M_enc_dm_DMO = dmo(box, rs)
    
    Omega_M = sim_params[box,0]
    M_enc_dm_DMO = np.log10( 10**(M_enc_dm_DMO) * (Omega_M - Omega_b)/Omega_M )
    
    r200_index = np.argmin(np.abs(rs - r200))
    
    ## Get fbaryon and fnorm
    f_norm = (10**M_enc_dm[r200_index] + 10**M_enc_stars[r200_index]) / 10**M_enc_dm_DMO[r200_index]
    fb     = 10**M_enc_stars[r200_index] / (10**M_enc_dm_DMO[r200_index])
    
    ### Variable rename
    M_hydro_s  = M_enc_stars
    M_hydro_dm = M_enc_dm
    M_DMO      = M_enc_dm_DMO
    
    func_M_hydro_s = interp1d(rs,M_hydro_s,bounds_error=False,
                              fill_value=(M_hydro_s[0],M_hydro_s[-1]) )
    
    func_M_hydro_dm = interp1d(rs,M_hydro_dm,bounds_error=False,
                               fill_value=(M_hydro_dm[0],M_hydro_dm[-1]) )
    
    func_M_DMO = interp1d(rs,M_DMO,bounds_error=False,
                          fill_value=(M_DMO[0],M_DMO[-1]) )
    
    ### Get A and w for Gnedin+2004 fits
    def menc_AC(rs, A, w):
        func_rbar = lambda r: r200 * A * (r/r200)**w
        
        func_r_contract_G04 = lambda rf: rs*(
            10**func_M_DMO(func_rbar(rs)) * f_norm
        )/(
            10**func_M_DMO(func_rbar(rs)) * f_norm * (1-fb) + 10**func_M_hydro_s(func_rbar(rf))
        )
        
        rf_G04 = fixed_point(func_r_contract_G04,rs)
        
        func_M_DM_G04 = interp1d(rf_G04,10**M_DMO,bounds_error=False,
                                 fill_value=(10**M_DMO[0],10**M_DMO[-1]))        
        return np.log10(func_M_DM_G04(rs))
    
    try:
        
        initial_guess = [0.4, 0.45] 
        bounds = [(0.01, 0.01), (3.5, 3.5)]
        (A_guess, w_guess), cov = curve_fit(menc_AC, rs, M_hydro_dm, p0=initial_guess, bounds=bounds, maxfev=3000)
        A_err, w_err = np.sqrt(np.diagonal(cov))

        print(f'A_cf: {A_guess:0.3f} +/- {A_err:0.3f}')
        print(f'w_cf: {w_guess:0.3f} +/- {w_err:0.3f}')

    
    except RuntimeError:
        ## M(r)r = M(r)r fit
        func_r_contract = lambda rf: rs*(
            10**func_M_DMO(rs) * f_norm 
        )/(
            #10**func_M_hydro_dm(rf)+10**func_M_hydro_s(rf)
            10**func_M_DMO(rs)+10**func_M_hydro_s(rf)
        ) 
        rf = fixed_point(func_r_contract,rs)
        func_M_DM = interp1d(rf,M_DMO,bounds_error=False,
                             fill_value=(M_DMO[0],M_DMO[-1]))
        
        with h5py.File(out_file_name, 'r+') as f:
            this_box = f.create_group(f'box_{box:03d}')
            this_box.create_dataset('radius', data=rs)
            this_box.create_dataset('menc_dmo', data=M_enc_dm_DMO)
            this_box.create_dataset('menc_hydro_dm', data=M_enc_dm)
            this_box.create_dataset('menc_hydro_baryons', data=M_hydro_dm)
            this_box.create_dataset('menc_dmo_AC_pred', data=np.zeros(len(M_enc_dm)))
            this_box.create_dataset('menc_dmo_L_pred', data=func_M_DM(rs))
            this_box.create_dataset('A', data=np.nan)
            this_box.create_dataset('A_uncert', data=[np.nan,np.nan])
            this_box.create_dataset('w', data=np.nan)
            this_box.create_dataset('w_uncert', data=[np.nan,np.nan])
            this_box.create_dataset('r200', data=r200)
        continue
            
    ## Gnedin+(2004) fit
    func_rbar = lambda r: r200 * A * (r/r200)**w
    func_r_contract_G04 = lambda rf: rs*(
        10**func_M_DMO(func_rbar(rs)) * f_norm 
    )/(
        10**func_M_DMO(func_rbar(rs)) * f_norm * (1-fb) + 10**func_M_hydro_s(func_rbar(rf))
    )
    
    rf_G04 = fixed_point(func_r_contract_G04,rs)
    func_M_DM_G04 = interp1d(rf_G04,M_DMO,bounds_error=False,
                             fill_value=(M_DMO[0],M_DMO[-1]))
    
    ## M(r)r = M(r)r fit
    func_r_contract = lambda rf: rs*(
        10**func_M_DMO(rs) * f_norm 
    )/(
        #10**func_M_hydro_dm(rf)+10**func_M_hydro_s(rf)
        10**func_M_DMO(rs) * f_norm * (1-fb)+10**func_M_hydro_s(rf)
    )
    rf = fixed_point(func_r_contract,rs)
    func_M_DM = interp1d(rf,M_DMO,bounds_error=False,
                         fill_value=(M_DMO[0],M_DMO[-1]))
    
    ## Plot
#     plt.plot(rs, menc_AC(rs, A, w), label='Emcee', linestyle='--', color='red')
#     plt.plot(rs, menc_AC(rs, A_guess, w_guess), label='Non-linear Least Squares', linestyle='--', color='C4')
#     # plt.plot(rs, menc_AC(rs, A_lsq, w_lsq), label='Lsq', linestyle='--', color='C6')
#     # plt.plot(rs, menc_AC(rs, A_cost, w_cost), label='Cost', linestyle='--', color='C7')
#     plt.plot(rs, func_M_DM(rs), label='Blumenthal', linestyle='-.', color='C8')
#     # plt.plot(rs, x, label='Hydro DM', linestyle='-', color='k')
#     plt.plot(rs, M_hydro_dm, label='Hydro DM', linestyle='-', color='k')
#     plt.plot(rs, M_DMO , label='DMO', linestyle=':', color='b')
    
#     plt.axvline(r200, color='gray', alpha=0.5)
    
#     plt.legend(frameon=False,fontsize=12,loc='upper left')
    
#     plt.text(0.95,0.075,r"${\rm DREAMS~Galaxy~%04d}$" %box, transform=plt.gca().transAxes, ha='right')
    
#     plt.xscale('log')

#     plt.xlabel(r'${\rm Radius}~[{\rm kpc}]$')
#     plt.ylabel(r'$M_{\rm enc}~{[M_\odot]}$')
    
#     plt.savefig(f'./figs/AC_curves/box_{box:04d}.pdf', bbox_inches='tight')
#     plt.close()
    
    with h5py.File(out_file_name, 'r+') as f:
        this_box = f.create_group(f'box_{box:04d}')
        this_box.create_dataset('radius', data=rs)
        this_box.create_dataset('menc_dmo', data=M_enc_dm_DMO)
        this_box.create_dataset('menc_hydro_dm', data=M_enc_dm)
        this_box.create_dataset('menc_hydro_baryons', data=M_hydro_dm)
        this_box.create_dataset('menc_dmo_AC_pred', data=menc_AC(rs, A, w))
        this_box.create_dataset('menc_dmo_L_pred', data=func_M_DM(rs))
        this_box.create_dataset('A', data=A)
        this_box.create_dataset('A_uncert', data=A_uncert)
        this_box.create_dataset('w', data=w)
        this_box.create_dataset('w_uncert', data=w_uncert)
        this_box.create_dataset('r200', data=r200)
        this_box.create_dataset('renorm', data=1)
