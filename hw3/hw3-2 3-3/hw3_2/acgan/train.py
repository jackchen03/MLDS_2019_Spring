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
from torch.autograd import Variable
from torch import randn
from torch import randint
from model import Generator, Discriminator

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-e","--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("-b","--batch_size", type=int, default=32, help="batch size")
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

#######################################################################
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
adversarial_loss = torch.nn.BCELoss().cuda()
auxiliary_loss = torch.nn.MSELoss().cuda()

def update(iterator, generator, discriminator, optimizer_D, optimizer_G):
   
    for i, batch in enumerate(iterator):
        # Configure input
        real_imgs = batch[0].to(device)
        shuf_imgs = batch[1].to(device)
        text = batch[2].to(device)

        real_imgs = real_imgs.permute(0,3,1,2)
        shuf_imgs = shuf_imgs.permute(0,3,1,2)
        batch_size = real_imgs.shape[0]

        valid = torch.ones(batch_size,1, requires_grad=False).to(device)
        fake = torch.zeros(batch_size,1, requires_grad=False).to(device)
        # -----------------
        #  Train Generator
        # -----------------
        for _ in range(3):
          optimizer_G.zero_grad()

        # Sample noise as generator input
          noise = randn(batch_size, 100).to(device)
        #noise = Variable(FloatTensor(np.random.normal(0,1,(batch_size, 100))))
        #gen_text = Variable(LongTensor(np.random.randint(0,22,batch_size)))
          gen_text_1 = torch.zeros(batch_size,12,requires_grad=False).to(device)
          gen_text_2 = torch.zeros(batch_size,10,requires_grad=False).to(device)
          one_pos_for_gtext_1 = randint(0,12,(batch_size,))
          one_pos_for_gtext_2 = randint(0,10,(batch_size,))
          for idx_1,text_1 in enumerate(gen_text_1): text_1[one_pos_for_gtext_1[idx_1]] = 1
          for idx_2,text_2 in enumerate(gen_text_2): text_2[one_pos_for_gtext_2[idx_2]] = 1
          gen_text = torch.cat((gen_text_1,gen_text_2),1)
        
        # Generate a batch of images
          gen_imgs = generator(noise, gen_text)

        # Loss measures generator's ability to fool the discriminator
          validity, pred_label = discriminator(gen_imgs, gen_text)
        
          pred_label = Variable(pred_label)
        #gen_text = Variable(gen_text)
        #gen_text = gen_text.long()
          g_loss_1 = adversarial_loss(validity, valid)
        #print(gen_text)
        #g_loss_2 = auxiliary_loss(pred_label, torch.argmax(gen_text,1))
          g_loss_2 = auxiliary_loss(pred_label, gen_text)
        
          g_loss = 0.5*(g_loss_1+g_loss_2)
          g_loss.backward()
          optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(1):
          optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        #loss for real images  
 
          real_pred, real_aux = discriminator(real_imgs, text)
          d_real_loss_1 = adversarial_loss(real_pred, valid)
          d_real_loss_2 = auxiliary_loss(real_aux, text)
          d_real_loss = 0.5*(d_real_loss_1 + d_real_loss_2)
        
        #loss for fake image
          fake_pred, fake_aux = discriminator(gen_imgs.detach(), text)
          d_fake_loss_1 = adversarial_loss(fake_pred, fake)  
          d_fake_loss_2 = auxiliary_loss(fake_aux, text)
          d_fake_loss = 0.5*(d_fake_loss_1 + d_fake_loss_2)
      
        #total d_loss
          d_loss = (d_real_loss + d_fake_loss) / 2

          d_loss.backward()
          optimizer_D.step()
        print(
              " [Batch %d/%d] [D loss: %f] [G loss: %f]"
              % ( i, len(iterator), d_loss.item(), g_loss.item()),
              end = '\r'
        )


#######################################################################

train_df = pd.read_pickle('../train_img.pkl')
real_img_arr = np.concatenate(train_df.values[0],axis=0)[:]
shuf_img_arr = real_img_arr
np.random.shuffle(shuf_img_arr)


# ITER_NUM = len(real_img) // BATCH_SIZE
# print(len(real_img_arr))
# print(real_img_arr[0])
# print(real_img_arr[0].shape)

tags_df = pd.read_csv('../extra_data/tags.csv')
tags = tags_df['attr']
tags_list = []
tags_num_list = []
for tag in tags:
    tag = tag[1]

for tag in tags:
    tags_list.append(tag.split())

# ‘color hair’
# 'orange hair',  0  ,'white hair',  1, 'aqua hair', 2, 'gray hair', 3
#  'green hair', 4, 'red hair', 5, 'purple hair', 6, 'pink hair', 7
#   'blue hair', 8, 'black hair', 9, 'brown hair', 10, 'blonde hair', 11
# ‘color eyes’
#  'black eyes', 12, 'orange eyes', 13
#  'pink eyes', 14, 'yellow eyes', 15, 'aqua eyes', 16, 'purple eyes', 17
#  'green eyes', 18, 'brown eyes', 19, 'red eyes', 20, 'blue eyes', 21
#  two hot 

for tag_list in tags_list:
    if(tag_list[0] == 'orange'): num = 0
    elif(tag_list[0] == 'white'): num = 1
    elif (tag_list[0] == 'aqua'): num = 2
    elif (tag_list[0] == 'gray'): num = 3
    elif (tag_list[0] == 'green'): num = 4
    elif (tag_list[0] == 'red'): num = 5
    elif (tag_list[0] == 'purple'): num = 6
    elif (tag_list[0] == 'pink'): num = 7
    elif (tag_list[0] == 'blue'): num = 8
    elif (tag_list[0] == 'black'): num = 9
    elif (tag_list[0] == 'brown'): num = 10
    elif (tag_list[0] == 'blonde'): num = 11

    if(tag_list[2] == 'black'): num_2 = 12
    elif(tag_list[2] == 'orange'): num_2 = 13
    elif(tag_list[2] == 'pink'): num_2 = 14
    elif(tag_list[2] == 'yellow'): num_2 = 15
    elif(tag_list[2] == 'aqua'): num_2 = 16
    elif(tag_list[2] == 'purple'): num_2 = 17
    elif(tag_list[2] == 'green'): num_2 = 18
    elif(tag_list[2] == 'brown'): num_2 = 19
    elif(tag_list[2] == 'red'): num_2 = 20
    elif(tag_list[2] == 'blue'): num_2 = 21

    num_list = np.zeros(22)
    num_list[num] = 1
    num_list[num_2] = 1
    tags_num_list.append(num_list)

real_img_tensor = torch.from_numpy(real_img_arr).float()
shuf_img_tensor = torch.from_numpy(shuf_img_arr).float()
tags_tensor = torch.Tensor(tags_num_list)
torch_dataset = Data.TensorDataset(real_img_tensor, shuf_img_tensor, tags_tensor)

loader = Data.DataLoader( 
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 2 
)

G = Generator().cuda()
D = Discriminator().cuda()

optimizer_g = optim.Adam(G.parameters(),lr= LR*0.5, betas = (0.5, 0.99))
optimizer_d = optim.Adam(D.parameters(),lr= LR, betas = (0.5, 0.99))


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
