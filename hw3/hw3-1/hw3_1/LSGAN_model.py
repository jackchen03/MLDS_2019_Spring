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
        x = self.sig(x)
        return x

adversarial_loss = torch.nn.BCELoss()

def update_g(batch_size, iter_num, g_model, d_model, g_optimizer):
    g_model.train()
    epoch_loss = 0
    valid = torch.ones(batch_size,1).to(device)
    for i in range(iter_num):
        message = "g_batch" + str(i) + " starts"
        print(message,end="\r")

        g_optimizer.zero_grad()
        noise = randn(batch_size, 100).to(device)

        output = g_model(noise)
        #g_loss = torch.mean(torch.log(1 - d_model(output)))
        #print(output[0])

        g_loss = adversarial_loss(d_model(output),valid)
        #score_log = torch.log(score+1e-10)
        #loss = (-1)*torch.sum(score_log) / batch_size
        print("g_loss: {}".format(g_loss))
        g_loss.backward()
        g_optimizer.step()
        epoch_loss += g_loss.item()
        #print(g_loss.item())
    train_loss = epoch_loss / iter_num
    print('\n Train set: Average loss: {:.7f}'.format(train_loss))  
    return train_loss

def eval_g(noise, generator):
    generator.eval()
    img = generator(noise).permute(0,2,3,1)
    return img
    
def update_d(iterator, generator, discriminator, d_optimizer):
    discriminator.train()
    epoch_loss = 0
    for i,batch in enumerate(iterator):
        message = "d_batch" + str(i) + " starts"
        print(message, end = "\r")
        src = batch.to(device)
        batch_size = src.shape[0]

        valid = torch.ones(batch_size,1).to(device)
        fake = torch.zeros(batch_size,1).to(device)
        noise = randn(batch_size, 100).to(device)

        d_optimizer.zero_grad()

        real_images = src.permute(0,3,1,2)
        fake_images = generator(noise).detach()

        #print("REAL: {}".format(real_images[0]))
        #print("FAKE: {}".format(fake_images[0]))
    
        #d_loss = -1*(torch.mean(torch.log(discriminator(real_images)) + torch.log(1 - discriminator(fake_images))))    
        real_loss = adversarial_loss(discriminator(real_images),valid)
        fake_loss = adversarial_loss(discriminator(fake_images),fake)
        d_loss = (real_loss + fake_loss)
        #print("real_loss: {}".format(real_loss))
        #print("fake_loss: {}".format(fake_loss))
        print("d_loss: {}".format(d_loss))

        #ones = torch.ones(batch_size, 1).to(device)
        #loss_in_batch = torch.log(real_predicts+1e-10) + torch.log(1 - fake_predicts+1e-10)
        #d_loss = (-1)*torch.sum(loss_in_batch) / batch_size
        #d_loss = torch.sum(d_loss)
        d_loss.backward()
        d_optimizer.step()
        epoch_loss += d_loss.item()
    train_loss = epoch_loss / len(iterator.dataset)
    print('\n Train set: Average loss: {:.5f}'.format(train_loss))

    return train_loss

def update(iterator, generator, discriminator, optimizer_D, optimizer_G):

    for i, batch in enumerate(iterator):
        # Configure input
        real_imgs = batch.permute(0,3,1,2).to(device)
        batch_size = batch.shape[0]

        valid = torch.ones(batch_size,1, requires_grad=False).to(device)
        fake = torch.zeros(batch_size,1, requires_grad=False).to(device)
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        noise = randn(batch_size, 100).to(device)

        # Generate a batch of images
        gen_imgs = generator(noise)

        # Loss measures generator's ability to fool the discriminator
        #g_loss = adversarial_loss(discriminator(gen_imgs),valid)
        g_loss = 0.5*torch.mean((1 - discriminator(gen_imgs))**2)

        g_loss.backward(retain_graph=True)
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        #real_loss = adversarial_loss(discriminator(real_imgs), valid)
        #fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        #d_loss = (real_loss + fake_loss) / 2
        d_loss = 0.5 * torch.mean((discriminator(real_imgs)-1)**2) + 0.5* torch.mean( (discriminator(gen_imgs.detach()))**2)

        d_loss.backward()
        optimizer_D.step()
        print(
            " [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % ( i, len(iterator), d_loss.item(), g_loss.item()),
            end = '\r'
        )
