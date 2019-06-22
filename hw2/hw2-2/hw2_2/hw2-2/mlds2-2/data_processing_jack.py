import os 
import pandas as pd 

#training testing data path
TRAINING_PATH = 'clr_conversation.txt'
TESTING_PATH = 'test_input.txt'

#training data
train_list = []

with open(TRAINING_PATH, encoding='utf-8') as file:
#dialog
	dialogs = file.read().strip().split("+++$+++\n") 

	for idx_d, dialog in enumerate(dialogs):
		#sentences of each dialog
		sentences = dialog.strip().split('\n')

		for idx_s in range(len(sentences) - 1):
			eos = '<eos>'
			bos = '<bos>'
			letters_input = sentences[idx_s].strip().split(' ')
			letters_input = [bos] + letters_input + [eos]
			letters_response = sentences[idx_s + 1].strip().split(' ')
			letters_response = [bos] + letters_response + [eos]
			train_list.append([letters_input, letters_response])

train_dataframe = pd.DataFrame(train_list, columns = ["input", "response"])
train_dataframe.to_pickle('training_data.pkl')


####################################################################
# TESTING

#testing data
test_list = []

with open(TESTING_PATH, encoding='utf-8') as file:
	sentences = file.read().strip().split('\n')

	for idx_s in range(len(sentences) - 1):
		eos = '<eos>'
		bos = '<bos>'
		letters_input = sentences[idx_s].strip().split(' ')
		letters_input = [bos] + letters_input + [eos]
		letters_response = sentences[idx_s + 1].strip().split(' ')
		letters_response = [bos] + letters_response + [eos]
		test_list.append([letters_input, letters_response])

test_dataframe = pd.DataFrame(test_list, columns = ["input" , "response"])
test_dataframe.to_pickle('testing_data.pkl')




