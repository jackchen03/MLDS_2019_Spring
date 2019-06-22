import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import tensorflow as tf

import random
import math
import os

from data_preprocessing import pad_sequences

BATCH_SIZE = 64
TIME_STEP = 100
SENTENCE_MAX_LEN = 20
INPUT_SIZE = 4096
VOCAB_SIZE = 2880
HIDDEN_SIZE = VOCAB_SIZE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE)
    self.lstm2 = nn.LSTM(input_size = 2 * HIDDEN_SIZE, hidden_size = VOCAB_SIZE)
   
  def forward(self, input_seqs, pad_token):
    mid, (hidden1, cell1) = self.lstm1(input_seqs, None)
    pad_token = pad_token.repeat(input_seqs.shape[0], 80, 1)
    new_mid = torch.cat((pad_token, mid), 2)
    outputs, (hidden2, cell2) = self.lstm2(new_mid, (hidden1, cell1))
    
    return outputs, (hidden1, cell1), (hidden2, cell2)
  
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = VOCAB_SIZE, hidden_size = HIDDEN_SIZE)
    self.lstm2 = nn.LSTM(input_size = 2 * HIDDEN_SIZE, hidden_size = VOCAB_SIZE)
   
  def forward(self, input_word, hidden1, cell1, hidden2, cell2, pad_token):
    pad_token = pad_token.repeat(input_word.shape[0], 1, 1)
    mid, _ = self.lstm1(pad_token, (torch.unsqueeze(hidden1[:,-1,:], 1), torch.unsqueeze(cell1[:,-1,:], 1)))
    new_mid = torch.cat((input_word, mid), 2)
    outputs, _ = self.lstm2(new_mid, (torch.unsqueeze(hidden2[:,-1,:], 1), torch.unsqueeze(cell2[:,-1,:], 1)))
    
    return outputs
  
class Seq2Seq(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.encoder =  Encoder().to(device)
    self.decoder = Decoder().to(device)
    
  def forward(self, src, target, pad_token, bos_token, is_train):
    _, (hidden1, cell1), (hidden2, cell2) = self.encoder(src, pad_token)
    input = bos_token
    input = input.repeat(src.shape[0], 1, 1)
    outputs = []
    for t in range(SENTENCE_MAX_LEN):
      output = self.decoder(input, hidden1, cell1, hidden2, cell2, pad_token)
      outputs.append(output)
      if(is_train):
        input = torch.unsqueeze(target[:,t,:], 1)
      else:
        input = output
    return torch.cat(tuple(outputs),1)


def train(model, iterator, optimizer, loss_function, clip, vocab_size, pad_token, bos_token):
  model.train()
  epoch_loss = 0

  for i, batch in enumerate(iterator):
    src = batch[0].to(device)
    trg_pad = batch[1].to(device)
    #padding 0
    #one_hotted
    trg_one_hot_vec=[]
    for sentence in trg_pad:
       sentence = sentence.view(len(sentence),1)
       one_hot = torch.zeros(trg_pad.shape[1], vocab_size, dtype = torch.float32, device = torch.device(device)).scatter(1,sentence.long(),1)
       trg_one_hot_vec.append(one_hot)

    trg_one_hot_vec = torch.stack(trg_one_hot_vec) #tensor of one-hot encodings

    optimizer.zero_grad()

    output = model(src.float(), trg_one_hot_vec, pad_token, bos_token, True)
    output = output[:].view(-1, output.shape[-1])
    trg_pad = trg_pad[:].view(-1)
    
    loss = loss_function(output,trg_pad.long())

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip) #avoid explode

    optimizer.step()

    epoch_loss += loss.item()
 # print(len(iterator))
  train_loss = epoch_loss/len(iterator)
  print('Train set: Average loss: {:.5f}'.format(train_loss))


  return train_loss

#non one hot yet
def evaluate(model, iterator, loss_function, bos_token, vocab_size, pad_token):
  model.eval()
  epoch_loss = 0
  test_acc = 0
  with torch.no_grad():
    for i, (src,trg) in enumerate(iterator):
      src = src.to(device)
      trg = trg.to(device)

      trg_one_hot_vec=[]

      for sentence in trg:
        sentence = sentence.view(len(sentence),1)
        one_hot = torch.zeros(trg.shape[1], vocab_size, dtype = torch.float32, device = torch.device(device)).scatter(1,sentence.long(),1)
        trg_one_hot_vec.append(one_hot)

      output = model(src.float(),trg_one_hot_vec,pad_token, bos_token, False)
      output = output[:].view(-1, output.shape[-1])
      trg = trg[:].view(-1)

      loss = loss_function(output,trg.long())
      epoch_loss += loss.item()

      #accuracy
      #pred = output.argmax(dim = 1, keepdim = True)
      #test_acc += (pred.long()).eq(trg.view_as(pred)).sum().item()
  
    test_loss = epoch_loss/len(iterator.dataset)
    #print acc loss
    print('\nTest set:Average loss:{:.5f}, Accuracy:{}/{}({:.0f}%)\n'.format(epoch_loss,test_acc,len(iterator),100.*test_acc/len(iterator.dataset)))
  
  return test_loss









