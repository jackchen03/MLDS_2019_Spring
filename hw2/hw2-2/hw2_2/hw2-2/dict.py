import json
import numpy as np
from gensim.models.word2vec import Word2Vec 
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import torch.nn as nn

EMBED_SIZE = 250
def Create_Embed_layer() :
	model = Word2Vec.load('word2vec_min7.model')
	word2idx = {"<pad>": 0,"<unk>": 1 } # 初始化 [word : token] 字典，后期 tokenize 语料库就是用该词典。
	vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
	#print(len(vocab_list))

	# 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
	embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 2, model.vector_size))
	for i in range(len(vocab_list)):
	    word = vocab_list[i][0]
	    word2idx[word] = i + 2
	    embeddings_matrix[i + 2] = vocab_list[i][1]
	embeddings_matrix[1] = np.random.rand(EMBED_SIZE)

	embedding_layer = nn.Embedding(len(embeddings_matrix),EMBED_SIZE)
	return embedding_layer


model = Word2Vec.load('word2vec_min7.model')
word2idx = {"<pad>": 0, "<unk>": 1} # 初始化 [word : idx] 字典，后期 tokenize 语料库就是用该词典。
idx2word = {0: "<pad>", 1:  "<unk>"} # 初始化 [idx : word] 字典

vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
print(len(vocab_list)+ 2)
# pad_idx = 0, unk_idx = 1, bos_idx = 2, eos_idx = 3

# 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
for i in range(len(vocab_list)):
	word = vocab_list[i][0]
	idx2word[i+2] = word
	word2idx[word] = i + 2

# 把eos的idx換到3
org_eos_idx = word2idx['<eos>']
org_3_word = idx2word[3]
word2idx['<eos>'] = 3
word2idx[org_3_word] = org_eos_idx
idx2word[3] = '<eos>'
idx2word[org_eos_idx] = org_3_word

json1 = json.dumps(idx2word)
f = open("idx2word_min7.json","w")
f.write(json1)
f.close()

json2 = json.dumps(word2idx)
f = open("word2idx_min7.json","w")
f.write(json2)
f.close()
