# The DREAMS Project: Disentangling the Impact of Halo-to-Halo Variance and Baryonic Feedback on Milky Way Dark Matter Density Profiles

Based on the paper of the same name, Alex M. Garcia et al. (In Preparation). This repository contains the scripts, data, and figure associated with the paper. I provide a short description of the files/directories below for convenience. 

For any questions/comments/concerns please reach out to me at [alexgarcia@virginia.edu](mailto:alexgarcia@virginia.edu)

# Directories

- `Databases`/`Models`/`Outputs`: Should be empty on Github, but are holders for the Neural Network emulator databases, models, and Outputs
- `data`: Contains all of the data associated with this paper
- [`figs`]($Figures): Contains all of the pdfs associated with this paper
- [`calculation_scripts`]($Calculation-Scripts): Contains scripts used to generate much of the data associated with the findings of this work

## Main

Contains the scripts used to generate the figures

### Loading in DREAMS data

The scripts `tutorial.py` contains several key functions used to load in the DREAMS data and is called in several of the below functions

### Figures

PDF versions of all the figures available in `figs/`

- `Figure1.py`: Reads in precomuted density images and makes plot
- `Figure2.py`: Loads in density profiles and breaks them down into the DREAMS parameters
- `Figure3.py`: Loads in the emulator and makes predictions for gNFW normalized parameters
- `Figure4.py`: Loads in the emulator and makes predictions for gNFW shape parameters
- `Figure5.py`: Loads in the emulator and makes predictions for the mass growth at 0.01 R200
- `Figure6.py`: Loads in two emulators and makes two predecitions for stellar mass of central galaxy and supermassive black hole 
- `Figure7.py`: Same as Figure 2, but including the DMO simulations now
- `Figure8.py`: Same as Figures 2 and 7, but for adiabatic contraction calculation
- `Figure9.py`: Loads in data from Figure 8 and from [Hussein+2025](https://ui.adsabs.harvard.edu/abs/2025arXiv250114868H/abstract)

## Calculation Scripts

Many of the scripts used to either generate data or train the Neural Network emulators are contained within this directory

### Emulator

- `emulator_helpers.py`: script containing helper functions to do Neural Network Emulator

All of these files are predicated on `emulator_helpers.py` and emulate the specified relation conditioned up the 5 DREAMS parameters and halo mass of the host

- `Emulate_bh_mass.py`: Black hole mass of central supermassive black hole
- `Emulate_gNFW_norm.py`: gNFW normalization parameters ($\rho_s$ and $r_s$)
- `Emulate_gNFW.py`: gNFW shape parameters ($\alpha$, $\beta$, $\gamma$)
- `Emulate_mass_growth.py`: mass growth comparing to hydro and DMO simulation
- `Emulate_sm_mass.py`: stellar mass of central galaxy

### Other

- `Figure1_data_generation.py`: Load in DREAMS data and save kernel smoothed projections of the halos within R200
- `baryon_contract.py`: Load in Hydro and DMO sims and do [Gnedin+2004](https://ui.adsabs.harvard.edu/abs/2004ApJ...616...16G/abstract) adiabatic contraction calculation
- `get_gNFW_fits.py`: Perform MCMC fit for gNFW profiles
- `get_density.py`: Load in DREAMS data and get shell-averaged densities

