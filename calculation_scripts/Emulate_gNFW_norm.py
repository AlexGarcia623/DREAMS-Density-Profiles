#!/usr/bin/env python
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py, os, optuna, torch
from torch.utils.data import DataLoader
import tutorial
import emulator_helpers as em

## Simulation params
boxes  = np.arange(1024)
slopes = np.zeros(len(boxes))
snap   = 90
h      = 0.6909

halo_mass  = np.load('./data/halo_mass.npy')
def norm_params(params):
    nparams = params / np.array([1, 1, 3.6, 7.4, .1, 1.0])
    
    minimum = np.array([0.274, 0.780, 0.25, 0.5, 0.25, np.min(halo_mass)])
    maximum = np.array([0.354, 0.888,  4.0, 2.0, 4.0 , np.max(halo_mass)])

    nparams = (nparams - minimum)/(maximum - minimum)
    
    return nparams

param_path = '/standard/DREAMS/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'
sim_params = tutorial.get_params(param_path)
params     = tutorial.get_params(param_path)
params     = np.array(params)
sat_params = [] ## add in halo mass
for box in boxes:
    c = list(params[box])
    c.append(halo_mass[box])
    sat_params.append(c)
sat_params = np.array(sat_params)
nparams    = norm_params(sat_params)

## Model params
name         = "gNFW_norm" ## study name
seed         = 1234        ## random seed
sizes        = (.8,.1,.1)  ## train, validate, test split
batch_size   = 32          ## batch size
n_iter       = 10          ## n hyperparameter tuning iterations
n_trials     = 50          ## n optimization trials
nepoch       = 500         ## number of epochs
#### For fiducial model (for some reason does not work if I don't do this step)
lr           = 2e-3        ## learning rate
wd           = 6e-8        ## weight drop
n_layers     = 2           ## n hidden layers
out_features = [800, 900]  ## n ouput features in hidden layers
dr           = [0.2, 0.4]  ## droupout rate of hidden layers
#### i/o
input_size   = 6           ## simulation params
output_size  = 2           ## prediction size

### Get Densities ###
in_file = 'CDM_density_profiles.hdf5'

sat_params  = []
best_params = []#np.load('./data/gnfw_params.npy')

with h5py.File('./data/gnfw.hdf5', 'r') as file:
    for box in boxes:
        sat_params.append(nparams[box])
        best_params.append(list(file[f'box_{box:04d}']['params']))
    
best_params = np.array(best_params)
    
slopes = best_params[:,0:2]

sat_params = np.array(sat_params)
input_data = em.normalize_data(slopes)
#####################

## Split dataset
train_dataset = em.Dataset('train', seed, input_data, sat_params, sizes)
train_loader  = DataLoader(dataset=train_dataset, batch_size=batch_size)

valid_dataset = em.Dataset('valid', seed, input_data, sat_params, sizes)
valid_loader  = DataLoader(dataset=valid_dataset, batch_size=batch_size)

test_dataset = em.Dataset('test', seed, input_data, sat_params, sizes)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size)

## Setup model hyperparams
hparams = em.Hyperparameters(lr, wd, n_layers, out_features, dr, nepoch, output_size, name, input_size) 

if torch.cuda.is_available():
    device = torch.device('cuda') #gpu
else:
    device = torch.device('cpu')
print(device)

model = em.dynamic_model(n_layers, out_features, dr, input_size, output_size)
model.to(device)

print('First pass model (do not remove)')
train_losses, valid_losses = em.train_model(model, train_loader, valid_loader, hparams, device=device)

state_dict = torch.load("Models/"+hparams.name_model(), map_location=device)
model.load_state_dict(state_dict)

test_loss, err = em.test(test_loader, model, hparams, device)
valid_loss = np.min(valid_losses)
train_loss = train_losses[np.argmin(valid_losses)]
print('first pass model:',train_loss, valid_loss, test_loss)

def objective(trial):
    """
    This function is given to optuna to tune hyperparameters.
    Given the current trial, optuna will suggest new hyperparameters for this training session.
    
    Inputs
     - trial - an optuna object containing information on the current training session
     
    Returns
     - test_loss - the log loss from the testing dataset, used to compare trials and choose hyperparameters
    """
    hparams.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hparams.weight_decay = trial.suggest_float("weight_decay", 1e-8, 1, log=True)
    hparams.n_layers = trial.suggest_int("n_layers", 1, 5)
    
    of = []
    dr = []
    for i in range(hparams.n_layers):
        of.append(trial.suggest_int("n_units_l{}".format(i), 4, 1e3))
        dr.append(trial.suggest_float("dropout_l{}".format(i), 0.2, 0.8))
    
    hparams.out_features = of
    hparams.dropout_rate = dr
        
    model = em.dynamic_model(hparams.n_layers, hparams.out_features, hparams.dropout_rate,
                             hparams.input_size, hparams.output_size)
    model.to(device)
        
    train_dataset = em.Dataset('train', seed, input_data, sat_params, sizes)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=batch_size)

    valid_dataset = em.Dataset('valid', seed, input_data, sat_params, sizes)
    valid_loader  = DataLoader(dataset=valid_dataset, batch_size=batch_size)
    
    train_losses, valid_losses = em.train_model(model, train_loader, valid_loader, hparams, device)

    np.save(f"Outputs/train_loss_{hparams.name_model()}", train_losses)
    np.save(f"Outputs/valid_loss_{hparams.name_model()}", valid_losses)
    
    return np.mean(valid_losses[-10:])

for i in range(n_iter):
    if i < 9:
        continue
    _s_ = f'Starting iteration {i+1}/{n_iter}'
    print('')
    print('#'*(len(_s_)+8))
    print("##  "+_s_+ "  ##")
    print('#'*(len(_s_)+8))
    print('')
    
    storage  = f"sqlite:///{os.getcwd()}/Databases/optuna_{name}_{i:02d}"
    sampler  = optuna.samplers.TPESampler(n_startup_trials=n_trials//3)
    study    = optuna.create_study(study_name=name, sampler=sampler, storage=storage, load_if_exists=False)

    def wrapped_objective(trial):
        # Tag run number in model name to avoid overwrites
        hparams.name = f"{name}_run{i}_trial{trial.number}"
        return objective(trial)
    
    study.optimize(wrapped_objective, n_trials, gc_after_trial=True)