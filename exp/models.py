import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import trimap
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm_notebook as tqdm
import numpy as np
import time
import os

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def clear_dir(dir_name):
        os.system('rm -r '+dir_name)

def linear_transparent_eval(X_train, 
                            X_test, 
                            latent_dim, 
                            batch_size=4096, 
                            sigma=5,
                            max_epochs=2000,
                            early_stopping=3,
                            learning_rate=1e-3, 
                            dataset_name='_'):

    print('ILS evaluation in progress...')

    similarity_KLD = torch.nn.KLDivLoss(reduction='batchmean')
    
    def loss_function(X, Z, sigma=0.3):
        Sx = compute_similarity(X, sigma)
        Sz = compute_similarity(Z, sigma)#1/np.sqrt(2))
        loss = similarity_KLD(torch.log(Sx), Sz)
        return loss
    
    class LinearModel(nn.Module):
        def __init__(self, input_shape, latent_dim=2):
            super(LinearModel, self).__init__()

            # encoding components
            self.fc1 = nn.Linear(input_shape, latent_dim)

        def encode(self, x):
            x = self.fc1(x)
            return x

        def forward(self, x):
            z = self.encode(x)
            return z

    def compute_similarity(X,sigma=5):
        M = torch.exp((-torch.cdist(X,X)**2)/(2*sigma**2))
        return M / (torch.ones([M.shape[0],M.shape[1]])*(torch.sum(M, axis = 0)-1)).transpose(0,1)

    train_dataset = TensorDataset(torch.tensor(X_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_dataset = TensorDataset(torch.tensor(X_test).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    
    # Create Model
    model = LinearModel(X_train.shape[1], latent_dim=latent_dim)

    clear_dir('./models/weights')
    check_mkdir('./models/weights')

    model_params = list(model.parameters())
    optimizer = torch.optim.Adam(model_params, lr=learning_rate)

    # record training process
    epoch_train_losses = []
    epoch_test_losses = []

    #validation parameters
    epoch = 1
    best = np.inf

    # progress bar
    pbar = tqdm(bar_format="{postfix[0]} {postfix[1][value]:03d} {postfix[2]} {postfix[3][value]:.5f} {postfix[4]} {postfix[5][value]:.5f} {postfix[6]} {postfix[7][value]:d}",
                postfix=["Epoch:", {'value':0}, "Train Sim Loss", {'value':0}, "Test Sim Loss", {'value':0}, "Early Stopping", {"value":0}])

    # start training
    while epoch <= max_epochs:

        # ------- TRAIN ------- #
        # set model as training mode
        model.train()
        batch_loss = []

        for batch, (X_batch,) in enumerate(train_loader):
            optimizer.zero_grad()
            Z_batch = model(X_batch)  #
            loss  = loss_function(X_batch, Z_batch, sigma) 
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        # save result
        epoch_train_losses.append(np.mean(batch_loss))
        pbar.postfix[3]["value"] = np.mean(batch_loss)

        # -------- VALIDATION --------

        # set model as testing mode
        model.eval()
        batch_loss = []

        with torch.no_grad():
            for batch, (X_batch,) in enumerate(test_loader):
                Z_batch = model(X_batch)
                loss = loss_function(X_batch, Z_batch, sigma)
                batch_loss.append(loss.item())

        # save information
        epoch_test_losses.append(np.mean(batch_loss))
        pbar.postfix[5]["value"] = np.mean(batch_loss)
        pbar.postfix[1]["value"] = epoch

        if epoch_test_losses[-1] < best:
            wait = 0
            best = epoch_test_losses[-1]
            best_epoch = epoch
            torch.save(model.state_dict(), f'./models/weights/LinearTransparent.pt')
        else:
            wait += 1
        pbar.postfix[7]["value"] = wait
        if wait == early_stopping:
            break    
        epoch += 1
        pbar.update()
    
    model.load_state_dict(torch.load(f'./models/weights/LinearTransparent.pt'))
    with torch.no_grad():
        model.eval()
        Z_train = model(torch.tensor(X_train).float()).cpu().detach().numpy()
        Z_test = model(torch.tensor(X_test).float()).cpu().detach().numpy()
    
    torch.save(model.state_dict(), f'./models/{dataset_name}_LinearTransparent_latent_{latent_dim}.pt')
    return {'Z_train':Z_train, 'Z_test':Z_test, 'debug':[epoch_train_losses,epoch_test_losses]}

def vae_eval(X_train, 
             X_test, 
             latent_dim, 
             batch_size = 4096, 
             n_epochs = 1000,
             keep_prob = 1,
             early_stopping = 5,
             learning_rate = 1e-4,
             n_hidden = None,
             dataset_name='_'
             ):

    print('VAE evaluation in progress...')

    if n_hidden:
        n_hidden=n_hidden
    else:
        n_hidden = X_train.shape[1]//2
        
    dim_img = X_train.shape[1]
    # Prepare data
    train_dataset = TensorDataset(torch.tensor(X_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_dataset = TensorDataset(torch.tensor(X_test).float())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 
    
    #https://github.com/dragen1860/pytorch-mnist-vae
    class Encoder(nn.Module):
        def __init__(self, imgsz, n_hidden, n_output, keep_prob):
            super(Encoder, self).__init__()
            self.imgsz = imgsz
            self.n_hidden = n_hidden
            self.n_output = n_output
            self.keep_prob = keep_prob
            self.net = nn.Sequential(
                nn.Linear(imgsz, n_hidden),
                nn.ELU(inplace=True),
                nn.Dropout(1-keep_prob),
                nn.Linear(n_hidden, n_hidden),
                nn.Tanh(),
                nn.Dropout(1-keep_prob),
                nn.Linear(n_hidden, n_output*2)
            )
        def forward(self, x):
            mu_sigma = self.net(x)
            # The mean parameter is unconstrained
            mean = mu_sigma[:, :self.n_output]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + F.softplus(mu_sigma[:, self.n_output:])
            return mean, stddev

        def get_mean(self, x):
            mu_sigma = self.net(x)
            # The mean parameter is unconstrained
            mean = mu_sigma[:, :self.n_output]
            return mean

    class Decoder(nn.Module):
        def __init__(self, dim_z, n_hidden, n_output, keep_prob):
            super(Decoder, self).__init__()
            self.dim_z = dim_z
            self.n_hidden = n_hidden
            self.n_output = n_output
            self.keep_prob = keep_prob
            self.net = nn.Sequential(
                nn.Linear(dim_z, n_hidden),
                nn.Tanh(),
                nn.Dropout(1-keep_prob),
                nn.Linear(n_hidden, n_hidden),
                nn.ELU(),
                nn.Dropout(1-keep_prob),
                nn.Linear(n_hidden, n_output),
                nn.Sigmoid()
            )
        def forward(self, h):
            return self.net(h)

    def get_y(decoder, z):
        # decoding
        y = decoder(z)
        y = torch.clamp(y, 1e-8, 1 - 1e-8)
        return y

    def get_z(encoder, x):
        # encoding
        mu, sigma = encoder(x)
        # sampling by re-parameterization technique
        z = mu + sigma * torch.randn_like(mu)
        return z, mu, sigma

    # In denoising-autoencoder, x_target == x + noise, otherwise x_target == x
    def get_loss(encoder, decoder, x, x_target):
        # encoding
        mu, sigma = encoder(x)
        # sampling by re-parameterization technique
        z = mu + sigma * torch.randn_like(mu)
        # decoding
        y = decoder(z)
        y = torch.clamp(y, 1e-8, 1 - 1e-8)
        # loss
        #marginal_likelihood2 = torch.mean(torch.sum(x_target * torch.log(y) + (1 - x_target) * torch.log(1 - y), 1))
        marginal_likelihood = torch.mean(torch.sum(-F.binary_cross_entropy(y, x_target, reduction='none'),1))
        #print(marginal_likelihood2.item(), marginal_likelihood.item())

        KL_divergence = torch.mean(0.5 * torch.sum(
                                                torch.pow(mu, 2) +
                                                torch.pow(sigma, 2) -
                                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                                                , 1))
        ELBO = marginal_likelihood - KL_divergence
        loss = -ELBO
        return y, z, loss, marginal_likelihood, KL_divergence
    
    wait = 0
    best_loss = np.inf

    encoder = Encoder(dim_img, n_hidden, latent_dim, keep_prob)
    decoder = Decoder(latent_dim, n_hidden, dim_img, keep_prob)
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    train_losses = []
    test_losses = []
    batches = len(train_loader)

    for epoch in tqdm(range(n_epochs)):
        total_loss = 0
        # Loop over all batches
        encoder.train()
        decoder.train()
        for i, (batch_X,) in enumerate(train_loader):
            optimizer.zero_grad()
            y, z, tot_loss, loss_likelihood, loss_divergence = get_loss(encoder, decoder, batch_X, batch_X)
            tot_loss.backward()
            optimizer.step()
            current_loss = tot_loss.item()
            total_loss += current_loss
        #if epoch % 10 == 0:
        #    print(f"epoch {epoch+1}: \
        #        train_L_tot {tot_loss.item():.4f} \
        #        train_L_likelihood {loss_likelihood.item():.4f} \
        #        train_L_divergence {loss_divergence.item():.2f}")
        train_losses.append([tot_loss.item(),loss_likelihood.item(),loss_divergence.item()])
        val_losses = 0
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            for i, (batch_X,) in enumerate(test_loader):
                y, z, tot_loss, loss_likelihood, loss_divergence = get_loss(encoder, decoder, batch_X, batch_X) 
                val_losses += tot_loss
            # print cost every epoch
            #if epoch % 10 == 0:
            #    print(f"epoch {epoch+1}: \
            #        test_L_tot {tot_loss.item():.4f} \
            #        test_L_likelihood {loss_likelihood.item():.4f} \
            #        test_L_divergence {loss_divergence.item():.2f}")
            test_losses.append([tot_loss.item(),loss_likelihood.item(),loss_divergence.item()])

        if val_losses.item() < best_loss:
            wait = 0
            best_loss = val_losses.item()
            best_epoch = epoch
            torch.save(encoder.state_dict(),f'./models/weights/VAE_Encoder.pt')
            torch.save(decoder.state_dict(),f'./models/weights/VAE_Decoder.pt')
        else:
            wait += 1
        if wait == early_stopping:
            break 
        #if epoch % 10 == 0:
        #    print('----------------------------------')
        
    torch.save(encoder.state_dict(),f'./models/{dataset_name}_VAE_Encoder_latent_{latent_dim}.pt')
    torch.save(decoder.state_dict(),f'./models/{dataset_name}_VAE_Decoder_latent_{latent_dim}.pt')
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        Z_train = encoder.get_mean(torch.tensor(X_train).float()).cpu().detach().numpy()
        Z_test = encoder.get_mean(torch.tensor(X_test).float()).cpu().detach().numpy()
    
    return {'Z_train':Z_train, 'Z_test':Z_test, 'debug':[train_losses,test_losses]}
            

def pca_eval(X_train, X_test, latent_dim):
    print('pca evaluation in progress...')
    pca = PCA(n_components=latent_dim)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    return {'model':pca, 'Z_train':Z_train, 'Z_test':Z_test}

def tsne_eval(X, latent_dim, n_jobs=-1):
    print('tsne evaluation in progress...')
    model = TSNE(n_components=latent_dim,
                 init='random',
                 n_jobs=n_jobs)
    Z = model.fit_transform(X)
    return {'model':model, 'Z_train':Z}

def umap_eval(X, latent_dim, n_jobs=-1):
    print('umap evaluation in progress...')
    model = umap.UMAP(n_components=latent_dim, 
                      n_jobs=n_jobs)
    Z = model.fit_transform(X)
    return {'model':model, 'Z_train':Z}

def trimap_eval(X, latent_dim):
    print('trimap evaluation in progress...')
    model = trimap.TRIMAP(n_dims=latent_dim, verbose=False)
    Z = model.fit_transform(X)
    return {'model':model, 'Z_train':Z}
