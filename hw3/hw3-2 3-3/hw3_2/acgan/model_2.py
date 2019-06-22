import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
from torch import randn
import numpy as np
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(random.random() * 1000000)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.textemb = nn.Linear(22, 256)
        def batch1d(out_feat):
          layer = nn.BatchNorm1d(out_feat, 0.9)
          return layer
        def batch2d(out_feat):
          layer = nn.BatchNorm2d(out_feat, 0.9)
          return layer
        self.hidden1 = nn.Linear(356, 512 * 4 * 4)
        self.batch1 = batch1d(512 * 4 * 4)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1)
        self.batch2 = batch2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1)
        self.batch3 = batch2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.batch4 = batch2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding = 1)
        self.tanh = nn.Tanh()

    def forward(self, noise, text):
        text_emb = F.relu(self.textemb(text))
        #print(text_emb.shape)
        concat = torch.cat((noise, text_emb), 1)
        x = F.relu(self.hidden1(concat))
        x = F.relu(self.batch1(x))
        x = x.view(-1,512,4,4)   #BATCH_SIZE*512*4*4
        x = self.conv1(x)
        x = F.relu(self.batch2(x))
        x = self.conv2(x)
        x = F.relu(self.batch3(x))
        x = self.conv3(x)
        x = F.relu(self.batch4(x))
        x = self.conv4(x)
        x = self.tanh(x)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def batch2d(out_feat):
          layer = nn.BatchNorm2d(out_feat, 0.9)
          return layer
        self.textemb = nn.Linear(22, 256)
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.batch1 = batch2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.batch2 = batch2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1)
        self.batch3 = batch2d(512)
        self.conv5 = nn.Conv2d(768, 512, kernel_size = 1, stride = 1, padding = 0)
        self.hidden1 =  nn.Linear(4 * 4 * 512, 1)
        self.aux = nn.Linear(4*4*512, 22)
        self.soft_max = nn.Softmax()
        self.sig = nn.Sigmoid()
        
        #size of image
        #ds_size = 100
        self.adv_layer = nn.Sequential(nn.Linear(4*4*512,1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(4*4*512,22), nn.Softmax())

    def forward(self, img, text):
        text_emb = F.relu(self.textemb(text))
        tiled_emb = text_emb.repeat(4,4,1,1)
        #print(tiled_emb.shape)
        tiled_emb = tiled_emb.permute(2,3,0,1)  # batch_size*256*4*4
        #print(tiled_emb.shape)
        x = F.leaky_relu(self.conv1(img)) 
        x = self.conv2(x)  
        x = F.leaky_relu(self.batch1(x)) 
        x = self.conv3(x) 
        x = F.leaky_relu(self.batch2(x)) 
        x = self.conv4(x) 
        x = F.leaky_relu(self.batch3(x))  # batch_size*512*4*4
        #print(x.shape)
        #print(tiled_emb.shape)
        concat = torch.cat((x, tiled_emb), 1)  # batch_size*(512+256)*4*4
        
        concat = self.conv5(concat)  # batch_size*512*4*4
        flat = concat.view(concat.shape[0], -1)

        
        validity = self.adv_layer(flat)
        label = self.aux_layer(flat)
        return validity, label

def eval_g(noise, generator):
    generator.eval()
    tag_list = []
    for i in range(25):
        tag = np.zeros(22)
        if(i < 5): 
            tag[7] = 1
            tag[12] = 1
        elif(i < 10): 
            tag[9] = 1
            tag[17] = 1
        elif(i < 15): 
            tag[5] = 1
            tag[20] = 1
        elif(i < 20): 
            tag[2] = 1
            tag[18] = 1
        else: 
            tag[11] = 1
            tag[13] = 1
        tag_list.append(tag)
    tags = np.asarray(tag_list)
    tags = torch.from_numpy(tags).float().to(device)
    img = generator(noise, tags).permute(0,2,3,1)
    return img


#pink hair black eyes -> 7, 12
#black hair purple eyes-> 9, 17
#red hair red eyes -> 5, 20
#aqua hair green eyes -> 2, 18
#blonde hair orange eyes -> 11, 13
