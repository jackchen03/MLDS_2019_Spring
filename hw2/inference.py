import numpy as np
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
from s2s import Seq2Seq
from s2s import train
import data_preprocessing
from data_preprocessing import pad_sequences

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    np.random.seed(1000)

    with open('word2index.obj','rb') as file:
        word2index = pickle.load(file)
    with open('index2word.obj','rb') as file:
        index2word = pickle.load(file)

    video_feat_folder = 'MLDS_hw2_1_data/testing_data/feat/'

    video_feat_filenames = listdir(video_feat_folder)
    video_feat_filepaths = [(video_feat_folder + filename) for filename in video_feat_filenames]

    #remove '.npy' from the filenames
    test_video_IDs = [filename[:-4] for filename in video_feat_filenames]

    video_feat_dict = {}
    for filepath in video_feat_filepaths:
        video_feat = np.load(filepath)
        video_ID = filepath[: -4].replace(video_feat_folder, "")
        video_feat_dict[video_ID] = video_feat
    
    model = Seq2Seq()
    model.load_state_dict(torch.load("s2s_model_16.pkl", map_location=device))

    VOCAB_SIZE = 2880
    bos_idx = torch.zeros(1,1,dtype=torch.long,device=torch.device(device))

    with open("testset_output.txt",'w') as outfile:
        for video_ID, feat in video_feat_dict.items():
            feat = torch.tensor(feat).unsqueeze(0).float().to(device)
            #print("feat size: {}".format(feat.shape))
            output = model(feat,None,bos_idx,0,is_train=False)
            output = torch.squeeze(output,0)
            #print(video_ID)
            #for i in range(output.shape[0]):
            #    print(output[i])
            lst = [index2word[ind.item()] for ind in torch.argmax(output,dim=1) if ind.item() is not 0]
            #print(lst.size())
            out_sentence = " ".join(tuple(lst))
            outfile.write("{},{}\n".format(video_ID,out_sentence))
