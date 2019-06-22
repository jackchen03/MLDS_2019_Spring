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
        self.hidden1 = nn.Linear(100, 128 * 8 * 8)
        #self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        self.conv1 = nn.ConvTranspose2d(128, 128, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv3 = nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding = 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = x.view(-1,128,8,8)   #BATCH_SIZE*128*16*16
        #x = self.upsample(x)    #BATCH_SIZE*128*32*32
        x = F.relu(self.conv1(x)) #BATCH_SIZE*128*32*32
        #x = self.upsample(x)    #BATCH_SIZE*128*64*64
        x = F.relu(self.conv2(x)) #BATCH_SIZE*64*64*64
        x = self.tanh(self.conv3(x))  #BATCH_SIZE*3*64*64
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1)
        self.hidden1 =  nn.Linear(4 * 4 * 256, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x)) #64*64*32
        x = F.relu(self.conv2(x)) #64*64*64
        x = F.relu(self.conv3(x)) #64*64*128
        x = F.relu(self.conv4(x)) #64*64*256
        x = x.view(x.shape[0], -1)
        x = self.hidden1(x)
        #x = self.sig(x)
        return x

adversarial_loss = torch.nn.BCELoss()


def eval_g(noise, generator):
    generator.eval()
    img = generator(noise).permute(0,2,3,1)
    return img
    

def update(iterator, generator, discriminator, optimizer_D, optimizer_G):
    one = torch.FloatTensor([1])
    mone = one * -1
    one = one.to(device)
    mone = mone.to(device)
    for i, batch in enumerate(iterator):
        
        for p in discriminator.parameters():
          p.requires_grad = True

        iter_d_num = 1 if (i>0.5*len(iterator)) else 2
 
        for d_iter in range(iter_d_num):
          optimizer_D.zero_grad()
          
          for p in discriminator.parameters():
            p.data.clamp_(-0.015,0.015)
          
           
           # Configure input
          real_imgs = batch.permute(0,3,1,2).to(device)
          batch_size = batch.shape[0]
          
          d_loss_real = torch.mean(discriminator(real_imgs))
          d_loss_real.backward(one)

         
          #valid = torch.ones(batch_size,1, requires_grad=False).to(device)
          #fake = torch.zeros(batch_size,1, requires_grad=False).to(device)
          # -----------------
          #  Train Generator
          # -----------------

          #optimizer_G.zero_grad()

          # Sample noise as generator input
          noise = randn(batch_size, 100).to(device)

          # Generate a batch of images
          gen_imgs = generator(noise)
          
          d_loss_fake = torch.mean(discriminator(gen_imgs))
          d_loss_fake.backward(mone)

          d_loss = d_loss_fake - d_loss_real
          optimizer_D.step()
        for p in discriminator.parameters():
          p.requires_grad = False
       
        optimizer_G.zero_grad()
        noise = randn(batch_size, 100).to(device)
        fake_imgs = generator(noise)
        g_loss = torch.mean(discriminator(fake_imgs))
        g_loss.backward(one)
        optimizer_G.step()
        
       
          
          # Loss measures generator's ability to fool the discriminator
          #g_loss = adversarial_loss(discriminator(gen_imgs),valid)
          #g_loss = torch.mean(discriminator(gen_imgs))
          #g_loss = torch.mean(torch.log(1 - discriminator(gen_imgs)))

          #g_loss.backward(retain_graph=True)
          #optimizer_G.step()

          # ---------------------
          #  Train Discriminator
          # ---------------------

          #optimizer_D.zero_grad()

          # Measure discriminator's ability to classify real from generated samples
          #real_loss = adversarial_loss(discriminator(real_imgs), valid)
          #fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
          #d_loss = (real_loss + fake_loss) / 2
          #d_loss = torch.mean(discriminator(real_imgs)) - torch.mean(discriminator(gen_imgs.detach()))
        #d_loss = (-1) * torch.mean(torch.log(discriminator(real_imgs)+1e-10) + torch.log(1 - discriminator(gen_imgs.detach())+1e-10))

          #d_loss.backward()
          #optimizer_D.step()
          #for p in discriminator.parameters():
          #  p.data.clamp_(-0.01, 0.01)
        print(
           " [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % ( i, len(iterator), d_loss.item(), g_loss.item()),
            end = '\r'
        )
