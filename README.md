# The DREAMS Project: Disentangling the Impact of Halo-to-Halo Variance and Baryonic Feedback on Milky Way Dark Matter Density Profiles

Based on the project of the same name, Alex M. Garcia et al. (In Preparation). This repository contains the scripts, data, and figure associated with the paper. I provide a short description of the files/directories below for convenience. 


## Directories

- `Databases`/`Models`/`Outputs`: Should be empty on Github, but are holders for the Neural Network emulator databases, models, and Outputs
- `data`: Contains all of the data associated with this paper
- `figs`: Contains all of the pdfs associated with this paper

## Files

### Generate Figures

- `Figure1.py`: Reads in precomuted density images and makes plot (see `Figure1_data_generation.py` and `Figure1.pdf` in `/figs/`)
- `Figure2.py`: Loads in density profiles and breaks them down into the DREAMS parameters (see `Figure2.pdf` in `/figs/`)
- `Figure3.py`: Loads in the emulator and makes predictions for gNFW normalized parameters (see `Figure3.pdf` in `/figs/`)
- `Figure4.py`: Loads in the emulator and makes predictions for gNFW shape parameters (see `Figure4.pdf` in `/figs/`)
- `Figure5.py`: Loads in the emulator and makes predictions for the mass growth at 0.01 R200 (see `Figure5.pdf` in `/figs/`)
- `Figure6.py`: Loads in two emulators and makes two predecitions for stellar mass of central galaxy and supermassive black hole (see `Figure6.pdf` in `/figs/`) 

### Emulator

- `emulator_helpers.py`: script containing helper functions to do Neural Network Emulator

All of these files are predicated on `emulator_helpers.py` and emulate the specified relation conditioned up the 5 DREAMS parameters and halo mass of the host

- `Emulate_bh_mass.py`: Black hole mass of central supermassive black hole
- `Emulate_gNFW_norm.py`: gNFW normalization parameters (rho\_s and r\_s)
- `Emulate_gNFW.py`: gNFW shape parameters (alpha, beta, gamma)
- `Emulate_mass_growth.py`: mass growth comparing to hydro and DMO simulation
- `Emulate_sm_mass.py`: stellar mass of central galaxy


### Other

- `Figure1_data_generation.py`: Load in DREAMS data and save kernel smoothed projections of the halos within R200

