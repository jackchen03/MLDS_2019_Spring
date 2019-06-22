from baseline import detect
import os

PATH = 'faces/'

file_list = []
for filename in os.listdir(PATH):
    detection = detect(filename)
    if(detection) : 
        file_list.append(file_name)
        
