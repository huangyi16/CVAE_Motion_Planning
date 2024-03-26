import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

thread = 8 #设到10以下
torch.set_num_threads(int(thread))

import time

import sys
# sys.path.append('~/devspace/CVAE_Motion_Planning')
# sys.path.append('~/devspace/python/CVAE_Motion_Planning')
sys.path.append('/home/huangyi/devspace/python/CVAE_Motion_Planning')
from flows import iaf

class Encoder(nn.Module):
    def __init__(self, input_dim, c_dim, h_Q_dim, z_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.c_dim = c_dim
        self.h_Q_dim = h_Q_dim
        self.z_dim = z_dim

        # self.layers = nn.ModuleList()
        # self.layers.append(nn.Linear(self.input_dim+self.c_dim, self.h_Q_dim))
        # self.layers.append(F.relu())
        # self.layers.append(nn.Dropout(0.5))
        # self.layers.append(nn.Linear(self.h_Q_dim, self.h_Q_dim))
        # self.layers.append(F.relu())
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim+self.c_dim, self.h_Q_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.h_Q_dim, h_Q_dim),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(self.h_Q_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.h_Q_dim, self.z_dim)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=1)
        x = self.layers(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)

        return z_mu, z_logvar
    
class Decoder(nn.Module):
    def __init__(self, z_dim, h_P_dim, c_dim, X_dim):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.h_P_dim = h_P_dim
        self.X_dim = X_dim

        self.layers = nn.Sequential(
            nn.Linear(self.z_dim+self.c_dim, self.h_P_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.h_P_dim, self.h_P_dim),
            nn.ReLU(),
            nn.Linear(self.h_P_dim, self.X_dim)
        )

    def forward(self, z, c):
        z = torch.cat((z, c), dim=1)
        y = self.layers(z)

        return y

class CVAEIAF(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim, h_Q_dim, h_P_dim, n_made_blocks, n_hidden_in_made, hidden_size):
        super(CVAEIAF, self).__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.h_Q_dim = h_Q_dim
        self.h_P_dim = h_P_dim
        
        self.n_made_blocks = n_made_blocks
        self.n_hidden_in_made = n_hidden_in_made
        self.hidden_size = hidden_size

        self.encoder = Encoder(self.x_dim, self.c_dim, self.h_Q_dim, self.z_dim)
        self.decoder = Decoder(self.z_dim, self.h_P_dim, self.c_dim, self.x_dim)

        # self.flow_layers = []
        # for idx in range(self.flow_length):
        #     self.flow_layers.append(iaf.IAF(self.z_dim, self.activation))
        # self.flow_layers = nn.ModuleList(self.flow_layers)
        self.iaf_flow = iaf.IAF(self.z_dim, self.hidden_size, self.n_hidden_in_made, self.n_made_blocks)

    def forward(self, x, c):
        z_mu, z_logvar = self.encoder(x, c)
        eps = torch.randn_like(z_mu)
        z = z_mu + torch.exp(z_logvar / 2) * eps

        z0 = z
        log_abs_det_jac = torch.zeros((z0.shape[0],)).to(z.device)

        # for layer in self.flow_layers:
        #     z, log_det = layer(z)
        #     log_abs_det_jac += log_det
        z, log_abs_det_jac = self.iaf_flow.inverse(z)
        
        recon_x = self.decoder(z, c)

        loss, recon_loss, kld = self.loss_function(recon_x, x, z_mu, z_logvar, z0, z, log_abs_det_jac)
        return recon_x, loss
    
    def loss_function(self, recon_x, x, z_mu, z_logvar, z0, zk, log_abs_det_jac):
        recon_loss = 0.5 * F.mse_loss(
            recon_x,
            x,
            reduction='none'
        ).sum(dim=1)

        log_prob_z0 = (
            -0.5 * (z_logvar + torch.pow(z0 - z_mu, 2) / torch.exp(z_logvar))
        ).sum(dim=1)

        log_prob_zk = (-0.5 * torch.pow(zk, 2)).sum(dim=1)

        KLD = log_prob_z0 - log_prob_zk - log_abs_det_jac

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


mb_size = 512 # batch size
h_Q_dim = 512 # encoder hidden layers dimension
h_P_dim = 512 # decoder hidden layers dimension

c = 0
lr = 1e-4

dim = 7
data_elements = dim + dim * 2 + 4 * 3
z_dim = 32
X_dim = dim * 64
y_dim = dim * 64
c_dim = dim * 2 

n_made_blocks = 4
n_hidden_in_made = 6
hidden_size = 256

# read data from npy
# c_filename = '../dataset_c.npy'
# x_filename = '../dataset_x.npy'

# dataset_c = np.load(c_filename)
# dataset_x = np.load(x_filename)
# print(dataset_c.shape)
# print(dataset_x.shape)
data_path = '../franka-100000.npy'
dataset = np.load(data_path, allow_pickle=True)
dataset_c_pre = dataset[:, 0]
dataset_x_pre = dataset[:, 1]
dataset_c = []
dataset_x = []
for idx in range(dataset_c_pre.shape[0]):
    c_line = dataset_c_pre[idx]
    x_line = dataset_x_pre[idx]
    dataset_c.append([item for sub in c_line for item in sub])
    dataset_x.append([item for sub in x_line for item in sub])
dataset_c = np.array(dataset_c)
dataset_x = np.array(dataset_x)

# data = np.concatenate((dataset_x, dataset_c), axis=1)
# print(data.shape)

num_data = dataset_c.shape[0]

ratio_test_train = 0.9
num_train = int(num_data * ratio_test_train)

X_train = dataset_x[0: num_train, :]
c_train = dataset_c[:num_train, :]

X_test = dataset_x[num_train:, :]
c_test = dataset_c[num_train:, :]
num_test = c_test.shape[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CVAEIAF(X_dim, c_dim, z_dim, h_Q_dim, h_P_dim, n_made_blocks, n_hidden_in_made, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for it in range(500001):
    # Randomly generate batches
    indices = torch.randint(0, num_train, (mb_size,))
    X_mb = torch.tensor(X_train[indices], dtype=torch.float32).to(device)
    c_mb = torch.tensor(c_train[indices], dtype=torch.float32).to(device)
    optimizer.zero_grad()
    recon_batch, loss = model(X_mb, c_mb)
    # loss = model.loss_function(recon_batch, X_mb, z_mu, z_logvar)
    loss.backward()
    optimizer.step()

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4f}'.format(loss.item()))
        print()


torch.save(model.state_dict(), '../checkpoints/cvae_nf_franka.pth')