import argparse
import os

import numpy as np
import pandas as pd
import torch 
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import torchvision.datasets as dsets
import torch.utils.data as Data
import random
from model import Generator, Discriminator
from model import update

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-e","--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("-b","--batch_size", type=int, default=64, help="batch size")
parser.add_argument("-mn","--model_name", type=str, default="model", help="model name for saving")
parser.add_argument("-lr","--learning_rate", type=float, default=0.0002, help="learning rate")
opt = parser.parse_args()
print(opt)

model_folder_path = "models/"+opt.model_name+"/"
os.makedirs(model_folder_path,exist_ok=True)

# parameters
EPOCH = opt.epochs
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate
D_UPD_NUM = 1
G_UPD_NUM = 1

###################################################################################################################################

train_df = pd.read_pickle('train_img.pkl')
real_img = np.concatenate(train_df.values[0],axis=0)[:]
ITER_NUM = len(real_img) // BATCH_SIZE
# print(len(real_img_arr))
# print(real_img_arr[0])
# print(real_img_arr[0].shape)

real_img_tensor = torch.from_numpy(real_img).float()

loader = Data.DataLoader( 
    dataset = real_img_tensor, 
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2 
)

G = Generator().to(device)
D = Discriminator().to(device)

optimizer_g = optim.Adam(G.parameters(),lr= LR, betas = (0.5, 0.99))
optimizer_d = optim.Adam(D.parameters(),lr= LR, betas = (0.5, 0.99))
# optimizer_g = optim.RMSprop(G.parameters(), lr=LR)
# optimizer_d = optim.RMSprop(D.parameters(), lr=LR)

for i in range(EPOCH):
        #train
    print("start training epoch"+str(i))
  
    update(loader, G, D,optimizer_d, optimizer_g)
    #for j in range(D_UPD_NUM): 
    #    update_d(loader, G, D, optimizer_d)
    #for j in range(G_UPD_NUM):
    #    update_g(BATCH_SIZE, ITER_NUM, G, D, optimizer_g)
    if(i%5==4):
        torch.save(G.state_dict(), model_folder_path+'G'+ str(i)+'.pkl')
        torch.save(D.state_dict(), model_folder_path+'D'+ str(i)+'.pkl')
