import numpy as np
#import matplotlib.pyplot as plt
import json
import pickle
import os
from os import listdir
import random
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from s2s import Encoder
from s2s import Decoder
from s2s import Seq2Seq
from s2s import train
import data_preprocessing
from data_preprocessing import pad_sequences

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH = 3
LR = 0.01
CLIP = 15
DECODER_LEN = 20
BATCH_SIZE=16
NUMLABEL = 3

if __name__ == "__main__":
    np.random.seed(1000)

    #train
    video_feat_folder = 'MLDS_hw2_1_data/training_data/feat/'
    training_label_json_file = 'MLDS_hw2_1_data/training_label.json'

    #test
    test_feat_folder = 'MLDS_hw2_1_data/testing_data/feat/'
    test_label_json_file = 'MLDS_hw2_1_data/testing_label.json'

    #train
    video_feat_filenames = listdir(video_feat_folder)
    video_feat_filepaths = [(video_feat_folder + filename) for filename in video_feat_filenames]

    #test
    test_feat_filenames = listdir(test_feat_folder)
    test_feat_filepaths = [(test_feat_folder + filename) for filename in test_feat_filenames] 

    #train
    with open('video_IDs.obj','rb') as file:
        video_IDs = pickle.load(file)
    with open('word2index.obj','rb') as file:
        word2index = pickle.load(file)
    with open('index2word.obj','rb') as file:
        index2word = pickle.load(file)
    with open('video_caption_dict.obj','rb') as file:
        video_caption_dict = pickle.load(file)
    
    #test
    with open('video_IDs_test.obj','rb') as file:
        test_video_IDS = pickle.load(file)
    with open('word2index_test.obj','rb') as file:
        test_word2index = pickle.load(file)
    with open('index2word_test.obj','rb') as file:
        test_index2word = pickle.load(file)
    with open('video_caption_dict_test.obj','rb') as file:
        test_video_caption_dict = pickle.load(file)
 
    VOCAB_SIZE = 2880
        
    video_feat_dict = {}
    for filepath in video_feat_filepaths:
        video_feat = np.load(filepath)
        video_ID = filepath[: -4].replace(video_feat_folder, "")
        video_feat_dict[video_ID] = video_feat
    
    pre_x = list(video_feat_dict.values())
    x = torch.tensor(pre_x * NUMLABEL)
    pre_y = []
    for i in range(NUMLABEL):
        pre_y += [captions[random.randint(1, 10000) % len(captions)].split() for captions in video_caption_dict.values()]
    y = []
    for sentence in pre_y:
        tokens = [3 if word2index.get(word) == None else word2index[word] for word in sentence] + [2]
        y.append(tokens)
    y = pad_sequences(y,maxlen=DECODER_LEN,padding='post',truncating='post')
    y = torch.LongTensor(y)
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=2,
    )
    #enc = Encoder().to(device)
    #dec = Decoder().to(device)

    model = Seq2Seq()
    model.cuda()
    

    #count param
    def count_parameters(model):
      return sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(),lr= LR)
    loss_function = nn.CrossEntropyLoss()

    loss_list = []

    for i in range(EPOCH):
        #train
        print("start training epoch"+str(i))
        loss = train(model, loader, optimizer, loss_function, CLIP, i)
        loss_list.append(loss)

    torch.save(model.state_dict(), 's2s_model_16.pkl')
    loss_list = np.array(loss_list)
    np.save('loss_list.npy', loss_list)
    #plt.plot(loss_list)
    #plt.title('loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch num')
    #plt.savefig('model_loss.png')
    #plt.clf()
