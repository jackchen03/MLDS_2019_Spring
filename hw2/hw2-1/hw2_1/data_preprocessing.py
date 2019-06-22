import numpy as np
import os
import sys
from os import listdir
import json
import pickle
import io

# Referenced from https://github.com/keras-team/keras/blob/master/keras/preprocessing/sequence.py.
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
    #transpose -> (max_length, len(sequence))
    #x = x.transpose()
    return x

def build_dictionary(sentences,min_count):
    word_counts = {}
    sentence_count=0
    for sentence in sentences:
        sentence_count+=1
        for word in sentence.lower().split():
            word_counts[word] = word_counts.get(word,0)+1
    dictionary = [word for word in word_counts if word_counts[word]>=min_count]

    print("Filtered words from %d to %d with min count [%d]" % (len(word_counts),len(dictionary),min_count))

    index2word = {
        0:'<pad>',
        1:'<bos>',
        2:'<eos>',
        3:'<unk>'
    }
    word2index = {
        '<pad>':0,
        '<bos>':1,
        '<eos>':2,
        '<unk>':3
    }

    for index,word in enumerate(dictionary):
        index2word[index+4] = word
        word2index[word] = index+4

    return word2index, index2word, dictionary

def filter_token(string):
    filters = '!"#$%&()\'*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    for c in filters:
        string = string.replace(c,'')
    return string

if __name__=="__main__":
    video_feat_folder = sys.argv[1]
    training_label_json_file = sys.argv[2]

    video_feat_filenames = listdir(video_feat_folder)
    video_feat_filepaths = [(video_feat_folder + filename) for filename in video_feat_filenames]

    #remove '.npy' from the filenames
    video_IDs = [filename[:-4] for filename in video_feat_filenames]
    #print(video_IDs)

    video_feat_dict = {} #store video features
    for path in video_feat_filepaths:
        video_feat = np.load(path)
        video_ID = path[:-4].replace(video_feat_folder,"")
        video_feat_dict[video_ID] = video_feat

    #store video captions
    video_captions = json.load(open(training_label_json_file,'r'))
    video_caption_dict = {}
    caption_corpus = []

    for video in video_captions:
        filtered_captions = [filter_token(sentence.lower()) for sentence in video["caption"]]
        video_caption_dict[video["id"]] = filtered_captions
        print(video["caption"])
        print(filtered_captions)
        caption_corpus += filtered_captions


    word2index, index2word, dictionary = build_dictionary(caption_corpus,min_count=3)
    #print(video_caption_dict)

    with open('./word2index.obj','wb') as file:
        pickle.dump(word2index,file,protocol=pickle.HIGHEST_PROTOCOL)
    with open('./index2word.obj','wb') as file:
        pickle.dump(index2word,file,protocol=pickle.HIGHEST_PROTOCOL)
    with open('video_IDs.obj', 'wb') as file:
        pickle.dump(video_IDs,file,protocol=pickle.HIGHEST_PROTOCOL)
    with open('video_caption_dict.obj', 'wb') as file:
        pickle.dump(video_caption_dict,file,protocol=pickle.HIGHEST_PROTOCOL)
    with open('video_feat_dict.obj', 'wb') as file:
        pickle.dump(video_feat_dict,file,protocol=pickle.HIGHEST_PROTOCOL)
