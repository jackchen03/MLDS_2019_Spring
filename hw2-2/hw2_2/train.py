import numpy as np
import pandas as pd
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
import torch.utils.data as Data

from gensim.models.word2vec import Word2Vec 
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

import json
import random
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from model import Seq2Seq
from keras.layers import Embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBED_SIZE = 250
BATCH_SIZE = 256
EPOCH = 400
LR = 0.001
CLIP = 5
DECODER_LEN = 15
ENCODER_LEN = 15

def train(model, iterator, optimizer, loss_function, clip, epoch_num):
  model.cuda()
  model.train()
  epoch_loss = 0
  
  #print("training starts")
  for i, batch in enumerate(iterator):
    src = batch[0].to(device)
    trg_pad = batch[1].to(device)
    #print(src.shape)
    #padding 0
    message = "batch" + str(i) + " starts"
    print(message, end = "\r")

    optimizer.zero_grad()
    output = model(src.float(), trg_pad, epoch_num, True)
    output = output[:].view(-1, output.shape[-1])
    trg_pad = trg_pad[:].view(-1)

    #print("output size:{}".format(output.shape))
    #print("trg_pad size:{}".format(trg_pad.shape))
    loss = loss_function(output,trg_pad.long())
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip) #avoid explode

    optimizer.step()

    epoch_loss += loss.item()
    #print("batch ends", end = '\r')

  train_loss = epoch_loss/len(iterator.dataset)
  print('\n Train set: Average loss: {:.5f}'.format(train_loss))

  return train_loss

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

if __name__ == "__main__":
    # with open('idx2word_min7.json') as json_file:
    #     idx2word = json.load(json_file)
    # with open('word2idx_min7.json') as json_file:
    #     word2idx = json.load(json_file)
    # 
    # model = Seq2Seq()
    # model.cuda()

    # optimizer = optim.Adam(model.parameters(),lr= LR)
    # loss_function = nn.CrossEntropyLoss()


    loss_list = []
    # training_df = pd.read_pickle("training_data.pkl")

    # for i in range(EPOCH) :
    #     shuffle_df = training_df.sample(frac = 0.01)
    #     x = shuffle_df['input'].tolist()
    #     y = shuffle_df['response'].tolist()

    #     # converting word to idx
    #     x_list = []
    #     y_list = []

    #     for sen_idx in range(len(x)):
    #         idx_list_x = []
    #         unk_num_x = 0
    #         for word in x[sen_idx]:
    #             idx = word2idx.get(word, word2idx['<unk>'])
    #             if(idx == word2idx['<unk>']): unk_num_x +=  1
    #             idx_list_x.append(idx)

    #         unk_num_y = 0
    #         idx_list_y = []
    #         for word in y[sen_idx]:
    #             idx = word2idx.get(word, word2idx['<unk>'])
    #             if(word == word2idx['<unk>']): unk_num_y +=  1
    #             idx_list_y.append(idx)
    #         if(unk_num_x <= 2 and  unk_num_y <= 2): 
    #             x_list.append(idx_list_x)
    #             y_list.append(idx_list_y)

    #     # word 2 index conversion completed, start padding

    #     x_padded = pad_sequences(x_list,maxlen=ENCODER_LEN,padding='post',truncating='post')
    #     y_padded = pad_sequences(y_list,maxlen=DECODER_LEN,padding='post',truncating='post')
    #     # padding complete
    x_padded_df = pd.read_pickle("train_x.pkl")
    y_padded_df = pd.read_pickle('train_y.pkl')
    
    
    model = Seq2Seq()
    model.cuda()

    optimizer = optim.Adam(model.parameters(),lr= LR)
    loss_function = nn.CrossEntropyLoss()

    for i in range(EPOCH):
        x_rand_df = x_padded_df.sample(frac = 0.05)
        y_rand_df = y_padded_df.sample(frac = 0.05)

        x_padded = x_rand_df.values.tolist()
        y_padded = y_rand_df.values.tolist()
        print(x_padded[0])
        print(len(x_padded))
        print(y_padded[0])
        print(len(x_padded))

        x_padded = torch.LongTensor(x_padded)
        y_padded = torch.LongTensor(y_padded)
    
        torch_dataset = Data.TensorDataset(x_padded, y_padded)

        loader = Data.DataLoader(
            dataset = torch_dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers=2,
        )

        #train
        print("start training epoch"+str(i))
        loss = train(model, loader, optimizer, loss_function, CLIP, i)
        loss_list.append(loss)
        if (i % 20 == 19):
            torch.save(model.state_dict(), 'chatbot_model_rand_new_' + str(i) + '.pkl')

    loss_list = np.array(loss_list)
    np.save('loss_list_rand_new.npy', loss_list)
