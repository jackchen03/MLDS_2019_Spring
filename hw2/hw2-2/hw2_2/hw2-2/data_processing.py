import os 
import pandas as pd 
import random
import json
import numpy as np
#MAX_TRAINING_SIZE = 2**18

#training testing data path
#TRAINING_PATH = 'clr_conversation.txt'
QUESTION_PATH = 'question.txt'
ANSWER_PATH = 'answer.txt'
TESTING_PATH = 'test_input.txt'

DECODER_LEN = 15
ENCODER_LEN = 15
##training data
#train_list = []
#
#with open(TRAINING_PATH, encoding='utf-8') as file:
##dialog
#        dialogs = file.read().strip().split("+++$+++\n") 
#        is_max = False
#        for idx_d, dialog in enumerate(dialogs):
#                #if is_max:
#                    #break
#                #sentences of each dialog
#                sentences = dialog.strip().split('\n')
#
#                for idx_s, sentence in enumerate(sentences):
#                        eos = '<eos>'
#                        bos = '<bos>'
#                        letters = sentence.strip().split(' ')
#                        letters = [bos] + letters + [eos]
#                        train_list.append([letters, idx_d, idx_s])
#                        #if(len(train_list)>=MAX_TRAINING_SIZE):
#                           # is_max = True
#
#train_dataframe = pd.DataFrame(train_list, columns = ["text", "#dialog", "#sentence"])
#
#print("Shape_train:", train_dataframe.shape)
#
#print(train_dataframe.sample(5))
#
#train_dataframe.to_pickle('train.pkl')
#
#pickle_dataframe = pd.read_pickle('train.pkl')
#
#print(train_dataframe.equals(pickle_dataframe))
#
#
##testing data
#test_list = []
#
#with open(TESTING_PATH, encoding='utf-8') as file:
#        sentences = file.read().strip().split('\n')
#
#        for idx_s,sentence in enumerate(sentences):
#                eos = '<eos>'
#                bos = '<bos>'
#                letters = sentence.strip().split(' ')
#                letters = [bos] + letters + [eos]
#                test_list.append([idx_s, letters])
#
#test_dataframe = pd.DataFrame(test_list, columns = ["#sentence" , "text"])
#
#print("Shape_test:", test_dataframe.shape)
#
#print(test_dataframe.sample(5))
#
#test_dataframe.to_pickle('test.pkl')
#pickle_dataframe = pd.read_pickle('test.pkl')
#
#print(test_dataframe.equals(pickle_dataframe))


###############################################################################################################
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


#data
x = []
y = []

with open(QUESTION_PATH, encoding='utf-8') as file:
#dialog
        #is_max = False
        dialogs = file.read().strip().split("+++$+++\n") 
        for idx_d, dialog in enumerate(dialogs):
                #if is_max:
                   #break
                #sentences of each dialog
                sentences = dialog.strip().split('\n')

                for idx_s in range(len(sentences)):
                        eos = '<eos>'
                        bos = '<bos>'
                        letters_input = sentences[idx_s].strip().split(' ')
                        letters_input = [bos] + letters_input + [eos] 
                        x.append(letters_input)            

with open(ANSWER_PATH, encoding='utf-8') as file:
#dialog
        #is_max = False
        dialogs = file.read().strip().split("+++$+++\n")
        for idx_d, dialog in enumerate(dialogs):
                #if is_max:
                   #break
                #sentences of each dialog
                sentences = dialog.strip().split('\n')

                for idx_s in range(len(sentences)):
                        eos = '<eos>'
                        bos = '<bos>'
                        letters_input = sentences[idx_s].strip().split(' ')
                        letters_input = [bos] + letters_input + [eos]
                        y.append(letters_input)

with open('idx2word_min7.json') as json_file:  
    idx2word = json.load(json_file)
with open('word2idx_min7.json') as json_file:  
    word2idx = json.load(json_file)

# converting word to idx
x_list = []
y_list = []

for sen_idx in range(len(x)):
            idx_list_x = []
            unk_num_x = 0
            for word in x[sen_idx]:
                idx = word2idx.get(word, word2idx['<unk>'])
                if(idx == 1): unk_num_x +=  1
                idx_list_x.append(idx)

            unk_num_y = 0
            idx_list_y = []
            for word in y[sen_idx]:
                idx = word2idx.get(word, word2idx['<unk>'])
                if(idx == 1): unk_num_y +=  1
                idx_list_y.append(idx)
            if(unk_num_x == 0 and  unk_num_y == 0):
                x_list.append(idx_list_x)
                y_list.append(idx_list_y)

# word 2 index conversion completed, start padding


x_padded = pad_sequences(x_list,maxlen=ENCODER_LEN,padding='post',truncating='post')

y_padded = pad_sequences(y_list,maxlen=DECODER_LEN,padding='post',truncating='post')

print(x_padded[0][0])
print(len(x_padded))
print(y_padded[0])
print(len(y_padded))
sen_l = []
for idx in x_padded[0]:
    print(idx)
    word = idx2word[str(idx)]
    sen_l.append(word)
print(sen_l)

x_pd = pd.DataFrame(x_padded)
x_pd.to_pickle('train_x.pkl')
y_pd = pd.DataFrame(y_padded)
y_pd.to_pickle('train_y.pkl')

####################################################################
# TESTING

#testing data
# test_list = []
# 
# with open(TESTING_PATH, encoding='utf-8') as file:
#         sentences = file.read().strip().split('\n')
# 
#         for idx_s in range(len(sentences)):
#                 eos = '<eos>'
#                 bos = '<bos>'
#                 letters_input = sentences[idx_s].strip().split(' ')
#                 letters_input = [bos] + letters_input + [eos]
#                 test_list.append(letters_input)
# 
# test_dataframe = pd.DataFrame(test_list, columns = ['test_data' ])
# test_dataframe.to_pickle('testing_data.pkl')





