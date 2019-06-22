import pandas as pd
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
from keras.layers import Embedding
import numpy as np
from gensim.models import Word2Vec
import torch.nn as nn

def Embedding(embed_size):
	model = Word2Vec.load('word2vec.model')
	word2idx = {"<pad>": 0} # 初始化 [word : token] 字典，后期 tokenize 语料库就是用该词典。
	vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]

	# 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
	embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
	for i in range(len(vocab_list)):
	    word = vocab_list[i][0]
	    word2idx[word] = i + 1
	    embeddings_matrix[i + 1] = vocab_list[i][1] #unexpected keyword?

	EMBEDDING_DIM = embed_size #词向量维度
	embedding_layer = nn.embedding(len(embeddings_matrix),EMBEDDING_DIM, weights= [embeddings_matrix], trainable = False)

	return embedding_layer


#print vocab size
if __name__ == "__main__":
	Embedding(250)
	


