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

BATCH_SIZE = 16
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

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, batch_first = True)
    self.lstm2 = nn.LSTM(input_size =  HIDDEN_SIZE, hidden_size = HIDDEN_SIZE, batch_first = True)
   
  def forward(self, input_seqs):
    mid, (hidden1, cell1) = self.lstm1(input_seqs, None)
    #pad_token = pad_token.repeat(input_seqs.shape[0], 80, 1)
    #new_mid = torch.cat((pad_token, mid), 2)
    outputs, (hidden2, cell2) = self.lstm2(mid, None)
    
    return outputs, (hidden1, cell1), (hidden2, cell2)

class Attn():
    def __init__(self):
        self.hidden_size = HIDDEN_SIZE

    def forward(self, hidden, encoder_outputs):
        length = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)

        # Create variable to store attention energies
        attn_energies = torch.zeros(batch_size, length).cuda() # B x S

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(length):
                attn_energies[b, i] = self.score(hidden[:,b,:], encoder_outputs[b, i].unsqueeze(0))

        alpha = F.softmax(attn_energies, dim = 1).float()
        h = encoder_outputs.float()

        a = torch.zeros(batch_size, HIDDEN_SIZE).cuda()
        for i in range(h.shape[1]):
            temp = h[:,i,:].clone() * alpha[:,i].unsqueeze(1)
            a = a + temp
        return torch.unsqueeze(a, 1)

    def score(self, hidden, encoder_output):
        # hidden [1, HIDDEN_SIZE], encoder_output [1, HIDDEN_SIZE]
        energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
        return energy

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    
    self.lstm1 = nn.LSTM(input_size = HIDDEN_SIZE, hidden_size = HIDDEN_SIZE, batch_first = True)
    self.lstm2 = nn.LSTM(input_size = EMBED_SIZE+ HIDDEN_SIZE, hidden_size = HIDDEN_SIZE , batch_first = True)
    self.attn = Attn()
   
  def forward(self, input_embed, encoder_outputs, hidden1, cell1, hidden2, cell2):
    # pad_token = pad_token.repeat(input_word.shape[0], 1, 1)
    # input_embed = 1 X 512
    context_vector = self.attn.forward(hidden1, encoder_outputs)
    mid, (hidden_out_1, cell_out_1) = self.lstm1(context_vector, (hidden1, cell1))
    
    new_mid = torch.cat((input_embed, mid), 2)
    outputs, (hidden_out_2, cell_out_2) = self.lstm2(new_mid, (hidden2, cell2))

    return outputs, hidden_out_1, cell_out_1, hidden_out_2, cell_out_2
  
class Seq2Seq(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.encoder =  Encoder()
    self.decoder = Decoder()
    self.out_net = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE).cuda()
    
  def forward(self, src, target, bos_idx, epoch_num, is_train):
    encoder_outputs, (hidden1, cell1), (hidden2, cell2) = self.encoder(src)
    
    input = bos_idx
    embedding = nn.Embedding(num_embeddings = VOCAB_SIZE, embedding_dim = EMBED_SIZE, padding_idx = 0).cuda()
    input_emb = embedding(input)
    # 1 X 512


    outputs = []
    for t in range(SENTENCE_MAX_LEN):
        output, hidden1, cell1, hidden2, cell2 = self.decoder(input_emb, encoder_outputs, hidden1, cell1, hidden2, cell2)
        final_out = self.out_net(output )
      
        outputs.append(final_out)
        if(is_train):
            if(epoch_num > 5):
                teacher_force_prob = TEACHER_FORCE_PROB_2
            else :
                teacher_force_prob = TEACHER_FORCE_PROB
            n, p = 1, teacher_force_prob  # number of trials, probability of each trial
            teacher = np.random.binomial(n, p, 1)[0]
        else :
            teacher = 0

        if(teacher):
            input_emb = embedding(target[:,t].unsqueeze(1))
            # _, indices = torch.max(final_out, 2)
            # print(indices.shape)
        else:
            # argmax of output
            # final_out = 64 X 1 X 2880
            _, indices = torch.max(final_out, 2)
            input_emb = embedding(indices)
    # print("finish forward once")
    return torch.cat(tuple(outputs), 1)


def train(model, iterator, optimizer, loss_function, clip, epoch_num):
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
    #sentence = sentence.view(len(sentence), 1)
    #embedding = nn.Embedding(num_embeddings = 2880, embedding_dim = VOCAB_SIZE, padding_idx = 0)
    #embedding(sentence)

    #one_hotted
    # trg_one_hot_vec=[]
    # for sentence in trg_pad:
    #    sentence = sentence.view(len(sentence),1)
    #    one_hot = torch.zeros(trg_pad.shape[1], vocab_size, dtype = torch.float32, device = torch.device(device)).clone()
    #    one_hot = one_hot.scatter_(1,sentence.long(), 1)
    #    trg_one_hot_vec.append(one_hot)

    # trg_one_hot_vec = torch.stack(trg_one_hot_vec) #tensor of one-hot encodings

    optimizer.zero_grad()

    output = model(src.float(), trg_pad, bos_idx, epoch_num, True)
    output = output[:].view(-1, output.shape[-1])
    trg_pad = trg_pad[:].view(-1)
    
    loss = loss_function(output,trg_pad.long())

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip) #avoid explode

    optimizer.step()

    epoch_loss += loss.item()
    print("batch ends")

  train_loss = epoch_loss/len(iterator.dataset)
  print('\n Train set: Average loss: {:.5f}'.format(train_loss))


  return train_loss










