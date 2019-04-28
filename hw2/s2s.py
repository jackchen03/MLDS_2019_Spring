import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.transforms as transforms
#import tensorflow as tf
from torchvision import models

import random
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from data_preprocessing import pad_sequences

BATCH_SIZE = 32
TIME_STEP = 100
SENTENCE_MAX_LEN = 20
INPUT_SIZE = 4096
VOCAB_SIZE = 2880
HIDDEN_SIZE = 256
EMBED_SIZE = 256
# LSTM_IN_SIZE = 128
TEACHER_FORCE_PROB = 0.8
TEACHER_FORCE_PROB_2 = 0.2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, batch_first = True)
    self.lstm2 = nn.LSTM(input_size =  HIDDEN_SIZE, hidden_size = HIDDEN_SIZE, batch_first = True)
   
  def forward(self, input_seqs,hc_init):
    mid, (hidden1, cell1) = self.lstm1(input_seqs, hc_init)
    outputs, (hidden2, cell2) = self.lstm2(mid, hc_init) 
    #print("h1={}, c1={}, h2={}, c2={}".format(hidden1, cell1, hidden2, cell2))
    #print("h1={}".format(hidden1[0][0])) 
    return outputs, (hidden1, cell1), (hidden2, cell2)

class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()
        self.hidden_size = HIDDEN_SIZE

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.dot_score(hidden, encoder_outputs) #(BATCH_SIZE,80)
        return F.softmax(attn_energies, dim = 1).unsqueeze(1) #(BATCH_SIZE,1,80) (for multiplication)

    #def score(self, hidden, encoder_output):
        # hidden [1, HIDDEN_SIZE], encoder_output [1, HIDDEN_SIZE]
        #energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
        #return energy

    def dot_score(self, hidden, encoder_output):
        # hidden(out)  (BATCH_SIZE,1,256)
        # encoder(out) (BATCH_SIZE,80,256)
        return torch.sum(hidden*encoder_output,dim=2)

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = HIDDEN_SIZE, hidden_size = HIDDEN_SIZE, batch_first = True)
    self.lstm2 = nn.LSTM(input_size = EMBED_SIZE+ HIDDEN_SIZE, hidden_size = HIDDEN_SIZE , batch_first = True)
    self.attn = Attn()
   
  def forward(self, last_word, encoder_outputs, last_context, hidden1, cell1, hidden2, cell2):
    paddings = torch.zeros(last_word.shape[0],1,EMBED_SIZE,device=torch.device(device))#(BATCH_SIZE, 1, 256)
    #print(context_vector.shape)
    #h,c: (1,BATCH_SIZE,256)
    mid, (hidden_out_1, cell_out_1) = self.lstm1(paddings, (hidden1, cell1))
    new_mid = torch.cat((last_word, mid), 2) #(BATCH_SIZE, 1, 512)
    outputs, (hidden_out_2, cell_out_2) = self.lstm2(new_mid, (hidden2, cell2))

    attn_weights = self.attn(outputs, encoder_outputs) #(BATCH_SIZE,1,80)
    context = attn_weights.bmm(encoder_outputs)        #(BATCH_SIZE,1,EMBED_SIZE) 

    return outputs, context, hidden_out_1, cell_out_1, hidden_out_2, cell_out_2
  
class Seq2Seq(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder =  Encoder()
    self.decoder = Decoder()
    self.out_net = nn.Linear(EMBED_SIZE, VOCAB_SIZE)
    self.embedding = nn.Embedding(num_embeddings = VOCAB_SIZE, embedding_dim = EMBED_SIZE, padding_idx = 0)
    
  def forward(self, src, target, bos_idx, epoch_num, is_train=True, total_epoch=0):
    #randn_h = torch.randn(1,src.shape[0],EMBED_SIZE).to(device)
    #randn_c = torch.randn(1,src.shape[0],EMBED_SIZE).to(device)
    hc_init = None
    encoder_outputs, (hidden1, cell1), (hidden2, cell2) = self.encoder(src,hc_init)
    #print("h1={}".format(hidden1)) 
    input = bos_idx
    input_emb = self.embedding(input) #last word

    outputs = []
    context = torch.zeros(src.shape[0],1,EMBED_SIZE).to(device) #context vector at each time step
    for t in range(SENTENCE_MAX_LEN):
        #print(input_emb[0]) 
        output, context, hidden1, cell1, hidden2, cell2 = self.decoder(input_emb, encoder_outputs, context, hidden1, cell1, hidden2, cell2)
        #print(torch.cat((output,context),dim=2).shape)
        #final_out = self.out_net(torch.cat((output,context),dim=2))
        final_out = self.out_net(output)
        outputs.append(final_out)

        if(is_train):
            #if(epoch_num > 100):
            #    teacher_force_prob = TEACHER_FORCE_PROB_2
            #else :
            #    teacher_force_prob = TEACHER_FORCE_PROB
            n, p = 1, 1-epoch_num/total_epoch  # number of trials, probability of each trial
            teacher = np.random.binomial(n, p, 1)[0]
        else :
            teacher = 0
        #teacher=0
        if(teacher):
            input_emb = self.embedding(target[:,t].unsqueeze(1))
        else:
            # argmax of output
            # final_out = BATCH_SIZE X 1 X 2880
            _, indices = torch.max(final_out, 2)
            # indices = BATCH_SIZE X 1
            input_emb = self.embedding(indices)
    return torch.cat(tuple(outputs), 1)


def train(model, iterator, optimizer, loss_function, epoch_num, clip,index2word, total_epoch):
  model.cuda()
  model.train()
  epoch_loss = 0
  
  #print("training starts")
  for i, batch in enumerate(iterator):
    src = batch[0].to(device)
    trg_pad = batch[1].to(device)
    #padding 0
    message = "batch" + str(i) + " starts"
    print(message, end = "\r")
    bos_idx = torch.ones(src.shape[0],1,dtype=torch.long,device=torch.device(device))

    optimizer.zero_grad()

    output = model(src.float(), trg_pad, bos_idx, epoch_num, True,total_epoch)
    _, outmax = torch.max(output,2)#(BATCH_SIZE,20)
    line = " "
    for sentence in outmax:
        outwords = line.join([index2word[i.item()] for i in sentence if i.item()!=0])
        #print(outwords)
    output = output[:].view(-1, output.shape[-1])
    trg_pad = trg_pad[:].view(-1)
    
    loss = loss_function(output,trg_pad.long())

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip) #avoid explode

    optimizer.step()

    epoch_loss += loss.item()
    #print("batch ends", end = '\r')
    #print(len(iterator.dataset))
  train_loss = epoch_loss/len(iterator.dataset)
  print('\n Train set: Average loss: {:.5f}'.format(train_loss))


  return train_loss


