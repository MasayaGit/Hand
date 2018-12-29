
# coding: utf-8

# In[1]:


import cv2, os, copy,glob
from sklearn.externals import joblib
import numpy as np
from keras.models import load_model
from keras.models import model_from_json

path = path= os.getcwd()
path_HandTest = path + '/Data/HandTest'
files = glob.glob(path_HandTest + "/*.jpg")

model = load_model('Hand.h5')
labels = ["若者","大人","高齢者"]

p = [] # 予想結果
image_size = (32,32) #画像の学習サイズ

for f in files:
    print(f)
    im = cv2.imread(f)
    #色空間を変換してリサイズ
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, image_size)
    # 配列の変形とデータの正規化
    im = im.reshape(32,32,3).astype('float32')/255
    
    # 予測
    r = model.predict(np.array([im]),batch_size=32,verbose=1) # --- (*4)

    res = r[0]
    #結果　
    #インデックス番号、要素の順に取得。
    for i,acc in enumerate(res):
        print(labels[i], "=",int(acc * 100))
    print("-----")
    print("予測した結果＝",labels[res.argmax()])
    print()

#"1000000",70,"male","dark",0,0,"palmar left","Hand_0009400.jpg",0 高齢者"

#"3000000",19,"female","very fair",1,0,"dorsal left","Hand_0009531.jpg",0 "若者"


#"3000000",19,"female","very fair",1,0,"palmar left","Hand_0009543.jpg",0 "若者"

#"8000000",21,"male","medium",0,0,"palmar right","Hand_0009735.jpg",0 "若者"

#"2000000",75,"female","dark",0,0,"palmar right","Hand_0009485.jpg",0 "高齢者"

#"0001002",54,"male","medium",0,0,"dorsal left","Hand_0003127.jpg",0 "大人"


#"0001518",43,"male","fair",0,0,"dorsal right","Hand_0001790.jpg",0  "大人"





