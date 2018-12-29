
# coding: utf-8

# In[1]:


import os, glob
import csv


def get20imageNumber():

    path= os.getcwd()
    path_HandInfo = path + '/Data/HandInfo.csv'

    Data_20_imageNumber = []

    count = 0 # 全体のカウント
    data_20_count = 0
    #csvファイルを開く
    readfile = open(path_HandInfo ,'r',encoding='utf-8')

    #ファイルから一行ずつ読み込む
    for line in readfile:
        Arraytext = line.split(",")        
        number = int(Arraytext[1])
        if number >= 20 and number < 30:
            Data_20_imageNumber.append(count)
            data_20_count += 1
        count += 1
        
    return Data_20_imageNumber


