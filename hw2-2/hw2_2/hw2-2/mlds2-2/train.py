import numpy as np
import torch
import pandas as pd
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
from keras.layers import Embedding
from gensim.models import Word2Vec


import random
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from model import Seq2Seq

BATCH_SIZE = 32
EPOCH = 200
LR = 0.01
CLIP = 15
DECODER_LEN = 10
ENCODER_LEN = 10
VOCAB_SIZE= 198307
EMBED_SIZE = 250

def train(model, iterator, optimizer, loss_function, num, clip, epoch_num):
  model.cuda()
  model.train()
  epoch_loss = 0
  
  #print("training starts"
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

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
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

  training_df = pd.read_pickle("training_data.pkl")
  x = training_df['input'].tolist()
  y = training_df['response'].tolist()

  model = Word2Vec.load('word2vec.model')
  word_vectors = model.wv
 
  #word2idx dict ind2word
  idx2word = {0:"<pad>", 1:"<unk>"}
  word2idx = {"<pad>":0, "<unk>":1}
  #embedding layers
  embeddings_matrix = torch.zeros((len(model.wv.vocab.items())+2, model.vector_size))
  EMBEDDING_DIM = EMBED_SIZE

  vocab_list = [(k, model.wv[k]) for k,v in model.wv.vocab.items()]

  for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    idx2word[i+2] = word
    word2idx[word] = i+2
    embeddings_matrix[i+2] = torch.from_numpy(vocab_list[i][1])

  org_eos_idx = word2idx['<eos>']
  org_3_word = idx2word[3]
  word2idx['<eos>'] = 3
  word2idx[org_3_word] = org_eos_idx
  idx2word[3] = '<eos>'
  idx2word[org_eos_idx] = org_3_word

  
  embedding_layer = nn.Embedding(len(embeddings_matrix), EMBEDDING_DIM, _weight = embeddings_matrix)

  print (len(word_vectors.vocab))
  #unknown's index will be len(word_vectors.vocab)
  for idx, sentence in enumerate(x):
    x[idx] = [word2idx[word] if word2idx.__contains__(word) else word2idx['<unk>'] for word in sentence]

  for idy, sentence in enumerate(y):
    y[idy] = [word2idx[word] if word2idx.__contains__(word) else word2idx['<unk>'] for word in sentence]

  x = pad_sequences(x, maxlen=ENCODER_LEN, padding='post',truncating='post')
  y = pad_sequences(y, maxlen=DECODER_LEN, padding='post', truncating='post')

  x = torch.LongTensor(x)
  y = torch.LongTensor(y)
  torch_dataset = Data.TensorDataset(x, y)

  
  loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers=2,
  )

  model = Seq2Seq(VOCAB_SIZE, embedding_layer)
  model.cuda()

  optimizer = optim.Adam(model.parameters(),lr= LR)
  loss_function = nn.CrossEntropyLoss()

  loss_list = []

  for i in range(EPOCH):
    #train
    print("start training epoch"+str(i))
    loss = train(model, loader, optimizer, loss_function, CLIP, i)
    loss_list.append(loss)

  torch.save(model.state_dict(), 's2s_model.pkl')
  loss_list = np.array(loss_list)
  np.save('loss_list.npy', loss_list)
















