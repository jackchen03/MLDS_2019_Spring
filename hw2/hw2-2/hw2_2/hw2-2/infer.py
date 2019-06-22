import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from train import pad_sequences
from model import Seq2Seq
import json

ENCODER_LEN=15
#ENCODER_LEN=20
TESTING_PATH = 'test_input.txt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
        with open('idx2word_min7.json') as json_file:  
                idx2word = json.load(json_file)
        with open('word2idx_min7.json') as json_file:  
                word2idx = json.load(json_file)

        model = Seq2Seq().to(device)
        model.load_state_dict(torch.load('chatbot_model_ppt199.pkl', map_location=device))

        test_list = []

        with open(TESTING_PATH, encoding='utf-8') as file:
                sentences = file.read().strip().split('\n')

                for sen in sentences:
                        eos = '<eos>'
                        bos = '<bos>'
                        letters_input = sen.strip().split(' ')
                        letters_input = [bos] + letters_input + [eos]
                        idx_list = []
                        for word in letters_input:
                            idx = word2idx.get(word, word2idx['<unk>'])
                            idx_list.append(idx)
                        test_list.append(idx_list)
        test_list_padded = pad_sequences(test_list,maxlen=ENCODER_LEN,padding='post',truncating='post')

        with open("testset_output_ppt199.txt",'w') as outfile:
                for sentence in test_list:
                        sentence = torch.tensor(sentence).unsqueeze(0).float().to(device)
                        output = model(sentence,None,0,is_train=False)
                        output = torch.squeeze(output,0)

                        lst = [idx2word[str(ind.item())] for ind in torch.argmax(output,dim=1) if ind.item() is not 0]
                        #print(lst)
                        out_sentence = " ".join(tuple(lst))
                        outfile.write("{}\n".format(out_sentence))
