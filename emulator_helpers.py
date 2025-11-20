import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py, os, optuna, torch
from torch.utils.data import DataLoader
import tutorial

def norm_params(params):
    """
    This function normalizes the four simulation parameters (WDN, SN1, SN2, AGN).
    
    Inputs
     - params - an Nx4 array of simulation parameters
    
    Results
     - nparams - same as the input but now normalized and linearly sampled between 0 and 1
    """
    nparams = params / np.array([1, 1, 3.6, 7.4, .1])
    
    minimum = np.array([0.274, 0.780, 0.25, 0.5, 0.25])
    maximum = np.array([0.354, 0.888, 4.0, 2.0, 4.0])

    nparams = (nparams - minimum)/(maximum - minimum)
    
    return nparams

def normalize_data(data,log=True):
    """
    This funciton normalizes the input data for the emulator
    
    Inputs
     - data - an array of values measured from the simulations
     
    Returns
     - out - the normalized array such that the mean = 0 and std = 1
    """
    if log == True:
        out = np.log10(data + 1)
    else:
        out = data
    mean = np.mean(out, axis=0)
    std = np.std(out, axis=0)
    out = (out - mean) / std
    return out

def split_dataset(dataset, train_size, valid_size, test_size):
    """
    This function splits the simulations into training, validation, and testing sets. 
    The data are split at the simulation level so that the network cannot learn part of the parameter space it is tested on.
    
    Inputs:
     - dataset - pytorch_geometric Data objects; one for each simulation
     - train_size - the fractional proportion of training data (0,1)
     - valid_size - the fractional proportion of validation data (0,1)
     - test_size  - the fractional proportion of testing data (0,1)
     
    Returns:
     - train_dataset - pytorch_geometric Data objects randomly selected to be in the training set
     - valid_dataset - pytorch_geometric Data objects randomly selected to be in the validation set
     - test_dataset  - pytorch_geometric Data objects randomly selected to be in the testing set
    """
    np.random.shuffle(dataset)
    
    ndata = len(dataset)
    split_valid = int(np.floor(valid_size * ndata))
    split_test = split_valid + int(np.floor(test_size * ndata))
    
    valid_dataset = dataset[:split_valid]
    test_dataset = dataset[split_valid:split_test]
    train_dataset = dataset[split_test:]
    
    return train_dataset, valid_dataset, test_dataset


class Dataset():
    """
    This object is used to match the input data (simulation parameters) with the output data (MW satellite counts)
    so that the emulator can compare its predictions to the correct values during training.
    This object also splits the data into the necessary training, validation, or testing sets based on the mode passed
    
    Inputs
     - mode - either 'train', 'valid', or 'test', determines which simulations to contain in the object
     - seed - the numpy random number seed
     - data - the normalized output data that the emulator will predict
     - params - the normalized input data that will be given to the emulator
     - sizes - a tuple of floats between 0 and 1 determing the share of data in the training, validation, and testing sets
    """
    
    def __init__(self, mode, seed, data, params, sizes):
        
        train_size = sizes[0]
        valid_size = sizes[1]
        test_size  = sizes[2]
        
        np.random.seed(seed)
        
        idx_dict = {'train':0, 'valid':1, 'test':2}
        sim_nums = np.arange(len(data))
        split_idx = split_dataset(sim_nums, train_size, valid_size, test_size)
        split_idx = split_idx[idx_dict[mode]]
                
        split_data = data[split_idx]
        split_params = params[split_idx]
        
        self.size = len(split_data)
        self.input = torch.tensor(split_params, dtype=torch.float)
        self.output = torch.tensor(split_data, dtype=torch.float)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

    
def dynamic_model(n_layers, out_features, dr, input_size, output_size):
    """
    This function defines the architecture of the fully-connected neural network.
    The inputs to the model are the relavent hyperparameters and the size of the 
      inputs and outputs.
      
    Inputs
     - n_layers - the number of fully-connected layers
     - out_features - a list of the number of nodes in each layer
     - dr - a list of the dropout rates for each layer
     - input_size - an integer of the number of input parameters
     - output_size - an integer of the number of output parameters
     
    Results
     - model - the pytorch neural network
    """
    # define the tuple containing the different layers
    layers = []

    # get the hidden layers
    in_features = input_size
    for i in range(n_layers):
        layers.append(torch.nn.Linear(in_features, out_features[i]))
        layers.append(torch.nn.LeakyReLU(0.2))
        layers.append(torch.nn.Dropout(dr[i]))
        in_features = out_features[i]

    # get the last layer
    layers.append(torch.nn.Linear(in_features, output_size*2))

    # return the model
    return torch.nn.Sequential(*layers)


class Hyperparameters():
    """
    This object acts as a container for the hyperparameters that are used during training.
    This object is also used to name files that are stored during training and testing.
    """
    def __init__(self, lr, wd, nl, of, dr, ne, os, name, input_size):
        
        self.learning_rate = lr
        self.weight_decay = wd
        self.n_layers = nl
        self.out_features = of
        self.dropout_rate = dr
        self.n_epochs = ne
        self.output_size = os
        self.study_name = name
        self.input_size = input_size
        
    def __repr__(self):
        return f"lr {self.learning_rate:.2e}; wd {self.weight_decay:.2e}; nl {self.n_layers}; of {self.out_features[0]}; dr {self.dropout_rate[0]:.2e}"
    
    def name_model(self):
        return f"{self.study_name}_lr_{self.learning_rate:.2e}_wd_{self.weight_decay:.2e}_nl_{self.n_layers}_of_{self.out_features[0]}_dr_{self.dropout_rate[0]:.2e}"

def train(loader, model, hparams, optimizer, scheduler, device):
    """
    This function loops over all data in the training dataset, calculates the loss, and updates the network parameters appropriately.
    
    Inputs
     - loader - a pytorch_geometric DataLoader object containing the training dataset
     - model - the partially trained network object
     - hparams - a Hyperparameters object containing the hyperparameters to be used in this training session
     - optimizer - the pytorch optimizer used to update the network
     - scheduler - the pytorch scheduler used to vary the training rates / momentum
     
    Returns
     - loss - the average log loss from this epoch
    """
    model.train()

    loss_tot = 0
    for data in loader:  # Iterate in batches over the training dataset.
        x = data[0].to(device)
        y = data[1].to(device)
        
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        optimizer.zero_grad()  # Clear gradients.
        out = model(x)  # Perform a single forward pass.
        
        y_out, raw_err_out = out[:,:hparams.output_size], out[:,hparams.output_size:2*hparams.output_size]     # Take mean and standard deviation of the output
        err_out = torch.nn.functional.softplus(raw_err_out)  # ensures >0
        if (err_out < 0).any():
            print("Warning: negative standard deviation predicted!")
        
        # Compute loss as sum of two terms for likelihood-free inference
        loss_mse = torch.mean(torch.sum((y_out - y)**2., axis=1) , axis=0)
        loss_lfi = torch.mean(torch.sum(((y_out - y)**2. - err_out**2.)**2., axis=1) , axis=0)
        loss = torch.log(loss_mse) + torch.log(loss_lfi)

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        scheduler.step()
        loss_tot += loss.item()

    return loss_tot/len(loader)

def test(loader, model, hparams, device):
    """
    This function loops over all data in the given (validation or testing) dataset and calculates the loss. 
    The parameters of the model are not updated in this function.
    
    Inputs
     - loader - a pytorch_geometric DataLoader object containing the validation or testing dataset
     - model - the partially trained network object
     - hparams - a Hyperparameters object containing the hyperparameters to be used in this training session
     
    Returns
     - loss - the average log loss from this epoch
     - errs - the average absolute error from the network predictions
    """
    model.eval()

    trueparams = np.zeros((0,hparams.output_size))
    outparams = np.zeros((0,hparams.output_size))
    outerrparams = np.zeros((0,hparams.output_size))

    errs = []
    loss_tot = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():
            
            x = data[0].to(device)
            y = data[1].to(device)
            if y.dim() == 1:
                y = y.unsqueeze(1)

            out = model(x)  # Perform a single forward pass.

            # If cosmo parameters are predicted, perform likelihood-free inference to predict also the standard deviation
            y_out, raw_err_out = out[:,:hparams.output_size], out[:,hparams.output_size:2*hparams.output_size]     # Take mean and standard deviation of the output
            err_out = torch.nn.functional.softplus(raw_err_out)  # ensures >0
            if (err_out < 0).any():
                print("Warning: negative standard deviation predicted!")
            
            # Compute loss as sum of two terms for likelihood-free inference
            loss_mse = torch.mean(torch.sum((y_out - y)**2., axis=1) , axis=0)
            loss_lfi = torch.mean(torch.sum(((y_out - y)**2. - err_out**2.)**2., axis=1) , axis=0)
            loss = torch.log(loss_mse) + torch.log(loss_lfi)

            err = (y_out - y)#/data.y
            errs.append( np.abs(err.detach().cpu().numpy()).mean() )
            loss_tot += loss.item()

            # Append true values and predictions
            trueparams = np.append(trueparams, y.detach().cpu().numpy(), 0)
            outparams = np.append(outparams, y_out.detach().cpu().numpy(), 0)
            outerrparams  = np.append(outerrparams, err_out.detach().cpu().numpy(), 0)
                
    
    # Save true values and predictions
    np.save("Outputs/trues_"+hparams.name_model()+".npy",trueparams)
    np.save("Outputs/outputs_"+hparams.name_model()+".npy",outparams)
    np.save("Outputs/errors_"+hparams.name_model()+".npy",outerrparams)

    return loss_tot/len(loader), np.array(errs).mean(axis=0)


def train_model(model, train_loader, valid_loader, hparams, device):
    """
    This is the main loop for training the network. For each epoch, the network is given data from the training set to update its parameters and is then tested on the validation set to see if the model has improved.
    If the model has improved, the network parameters are saved in a file which can be reloaded later.
    
    Inputs
     - model - the instantiated and untrained network
     - train_loader - a DataLoader object containing the training dataset
     - valid_loader - a DataLoader object containing the validation dataset
     - hparams - a Hyperparameters object containing the hyperparameters to be used in this training session
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hparams.learning_rate, max_lr=1.e-3, cycle_momentum=False, step_size_up=500)
    
    train_losses, valid_losses = [], []
    valid_loss_min, err_min = 1000., 1000.
    
    for epoch in range(1, hparams.n_epochs+1):
        train_loss = train(train_loader, model, hparams, optimizer, scheduler, device)
        valid_loss, err = test(valid_loader, model, hparams, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # Save model if it has improved
        if valid_loss <= valid_loss_min:
            torch.save(model.state_dict(), "Models/"+hparams.name_model())
            valid_loss_min = valid_loss
            err_min = err
            print(f"Epoch {epoch:03d} Train loss {train_loss:.2e} Valid loss {valid_loss:.2e} Error: {err:.2e} (B)")
        else:
            if epoch % 50 == 0:
                print(f"Epoch {epoch:03d} Train loss {train_loss:.2e} Valid loss {valid_loss:.2e} Error: {err:.2e}")
            
    return train_losses, valid_losses
    
def denormalize_data(data, true, pred, err):
    """
    This funciton denormalizes the results from the trained network
    
    Inputs
     - data - the original data used to normalize the data
     - true - the correct parameter from the simulation
     - pred - the predicted parameter from the network
     - err  - the predicted parameter error from the network
     
    Returns
     - ntrue - the denormalized true array
     - npred - the denormalized pred array
     - nerr  - the denormalized err array
    """
    out = np.log10(data + 1)
    mean = np.mean(out, axis=0)
    std = np.std(out, axis=0)
    
    # Denormalize the true values
    dn_true = np.power(10, (true * std + mean))
    
    # Denormalize the predicted values
    dn_pred = np.power(10, (pred * std + mean))
    
    # Denormalize the error values (apply the same transformation as pred)
    dn_err = np.power(10, (pred * std + mean)) * err  # Adjusted formula

    return dn_true, dn_pred, dn_err
    
def load_model(hparams, device='cpu'):
    model = dynamic_model(
        hparams.n_layers,
        hparams.out_features,
        hparams.dropout_rate,
        hparams.input_size,
        hparams.output_size
    )
    model.load_state_dict(torch.load(f"Models/{hparams.name_model()}", map_location=device))
    model.to(device)
    model.eval()
    return model
    
if __name__ == "__main__":
    print('Hello World!')
    
    