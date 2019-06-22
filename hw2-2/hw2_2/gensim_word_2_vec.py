import pandas as pd
from gensim.models.word2vec import Word2Vec 

train_df = pd.read_pickle("train.pkl")
test_df = pd.read_pickle("test.pkl")


#show five data from head (default)
#print(train_df.head())

#sample(frac=1) -> shuffle data
corpus = pd.concat([train_df.text, test_df.text]).sample(frac = 1)
#print(corpus.head())

#put it in Word2Vec
#adjust size iter sg=0 use skip-gram(low freq good) sg=1(use CBOW) ,window: use+-1 letter to predict 

#train 

#model = Word2Vec(corpus, size = 250, iter = 10, sg=0, window=3, min_count=7, max_vocab_size=None, workers=3, min_alpha=0.0001, hs=0, negative=5, batch_words=10000)
model.save('word2vec_min7.model')
#window = 3 word2vec
#window = 2 word2vec_window

#load
model = Word2Vec.load('word2vec_min7.model')

#tcheck trained model
def most_similar(w2v_model, words, topn=10):
	similar_df = pd.DataFrame()
	for word in words:
		try:
			similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn), columns=[word, 'cos'])
			similar_df = pd.concat([similar_df, similar_words], axis=1)
		except:
			print(word, "not found in Word2Vec model!")
	return similar_df


print(most_similar(model, ['行動','失敗','細節','好好','真是']))
