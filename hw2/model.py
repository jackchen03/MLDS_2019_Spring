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
TEACHER_FORCE_PROB_2 = 0.4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
  
class Seq2Seq(nn.Module):
  def __init__(self,vocab_size=VOCAB_SIZE):
    super().__init__()
    self.vocab_size = vocab_size
    self.encoder = nn.LSTM(input_size = """""", hidden_size = HIDDEN_SIZE, batch_first = True)
    self.decoder = nn.LSTM(input_size = EMBED_SIZE+HIDDEN_SIZE, hidden_size = HIDDEN_SIZE, batch_first = True)
    self.out_net = nn.Linear(HIDDEN_SIZE, self.vocab_size)
    self.embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = EMBED_SIZE, padding_idx = 0)
    self.attn = Attn()
  def forward(self, src, target, epoch_num, is_train):

    encoder_outputs, (h,c) = self.encoder(src)

    indices = torch.ones(src.shape[0],1,dtype=torch.long,device=torch.device(device)) #bos_idx
    #input_emb = self.embedding(input)
    #pad = torch.zeros(src.shape[0],1,HIDDEN_SIZE).to(device)
    context = torch.zeros(src.shape[0],1,HIDDEN_SIZE).to(device)
    
    outputs = []
    for t in range(SENTENCE_MAX_LEN):
        input_emb = torch.cat((self.embedding(indices),context),2)
        output, (h,c) = self.decoder(input_emb, (h,c))
        #attn
        attn_weights = self.attn(output, encoder_outputs) #(BATCH_SIZE,1,80)
        context = attn_weights.bmm(encoder_outputs)        #(BATCH_SIZE,1,EMBED_SIZE)
        final_out = self.out_net(output)
        outputs.append(final_out)
        if(is_train):
            indices = target[:,t].unsqueeze(1)
        else:
            _, indices = torch.max(final_out, 2)
        
        """if(is_train):
            if(epoch_num > 5):
                teacher_force_prob = TEACHER_FORCE_PROB_2
            else :
                teacher_force_prob = TEACHER_FORCE_PROB
            n, p = 1, teacher_force_prob  # number of trials, probability of each trial
            teacher = np.random.binomial(n, p, 1)[0]
        else :
            teacher = 0

        if(teacher):
            input_emb = self.embedding(target[:,t].unsqueeze(1))
            # _, indices = torch.max(final_out, 2)
            # print(indices.shape)
        else:
            # argmax of output
            # final_out = BATCH_SIZE X 1 X 2880
            _, indices = torch.max(final_out, 2)
            # indices = BATCH_SIZE X 1
            input_emb = self.embedding(indices)"""
    return torch.cat(tuple(outputs), 1)


def train(model, iterator, optimizer, loss_function, num, clip, index2word, epoch_num):
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

    optimizer.zero_grad()
    output = model(src.float(), trg_pad, epoch_num, True)
    output = output[:].view(-1, output.shape[-1])
    trg_pad = trg_pad[:].view(-1)
    
    loss = loss_function(output,trg_pad.long())
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip) #avoid explode

    optimizer.step()

    epoch_loss += loss.item()
    #print("batch ends", end = '\r')

  train_loss = epoch_loss/len(iterator.dataset)
  print('\n Train set: Average loss: {:.5f}'.format(train_loss))

  return train_loss
