from PIL import Image
import os
import numpy as np
import pandas as pd
import baseline

PATH = 'faces/'

pix_list = []
file_list = []
for filename in os.listdir(PATH):
    detection = baseline.detect(str(PATH + filename))
    if(detection) :
	      im = Image.open(str(PATH + filename))
	      im = im.resize((64,64))
	      arr = np.array(im)
	      pix_list.append(arr[np.newaxis,:])
	  
pix_arr = np.array(pix_list)

norm = 128*np.ones(pix_arr.shape)
#std = np.std(pix_arr)

pix_arr = (pix_arr - norm) / norm
pix_l = list(pix_arr)
x_list = []
x_list.append(pix_l)
pix_pd = pd.DataFrame(x_list)
pix_pd.to_pickle('train_img.pkl')
