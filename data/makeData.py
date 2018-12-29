
# coding: utf-8

# In[1]:


import cv2
import os, glob
import csv
import keras

from Data20_imageNumber import get20imageNumber

class makeData:
    
    def __init__(self):
        self.Data_20_imageNumber = get20imageNumber() #２０代の手の画像５００枚　番号
        self.count = 0 #全体のカウント
        self.Data_20_count = 0 #２０代のカウント
        self.Data_20_number = 200
    
    #２０代の時にtrueを返す それ以外ならfalse
    def count_image(self):
        if self.count == self.Data_20_imageNumber[self.Data_20_count]:
            self.Data_20_count += 1
            self.count += 1
            return True

        else:
            self.count += 1
            return False
        
   

    # 画像データを読み込んで配列に追加
    def appendX(self,path):

        x = [] # 画像データ

        image_size = (32,32) #画像の学習サイズ
        files = glob.glob(path + "/*.jpg")
        for f in files:
            if self.count_image():
                if  self.Data_20_count > self.Data_20_number:
                    continue
                im = cv2.imread(f)
                x.append(change_image(im))
                
            else:
                im = cv2.imread(f)
                x.append(change_image(im))
                x.append(change_image(im))
                img2,img3 = makeDataByFlip(im)
                x.append(change_image(img2))
                x.append(change_image(img3))
        return x

    

    # 画像データを読み込んで配列に追加
    def appendY(self,path):

        y = [] # ラベルデータ
        num_classes = 3 #分類数
        count = 0 # 20代のカウント
        #csvファイルを開く
        readfile = open(path ,'r',encoding='utf-8')

        
        #ファイルから一行ずつ読み込む
        for line in readfile:
            Arraytext = line.split(",")        
            number = int(Arraytext[1])

            if number < 20 :
                #ラベルデータをone-hot化
                y.append(keras.utils.to_categorical(0,num_classes))
                # データ増やした分考慮してさらに２回追加
                y.append(keras.utils.to_categorical(0,num_classes))
                y.append(keras.utils.to_categorical(0,num_classes))
                y.append(keras.utils.to_categorical(0,num_classes))
            #20代なら1に分類される
            if number >= 20 and number < 30:
                if count == self.Data_20_number:
                    continue
                y.append(keras.utils.to_categorical(0,num_classes))
                count += 1
            if number >= 30 and number < 40:
                y.append(keras.utils.to_categorical(1,num_classes))
                y.append(keras.utils.to_categorical(1,num_classes))
                y.append(keras.utils.to_categorical(1,num_classes))
                y.append(keras.utils.to_categorical(1,num_classes))

            if number >= 40 and number < 50:
                y.append(keras.utils.to_categorical(1,num_classes))
                y.append(keras.utils.to_categorical(1,num_classes))
                y.append(keras.utils.to_categorical(1,num_classes))
                y.append(keras.utils.to_categorical(1,num_classes))
            if number >= 50 and number < 60:
                y.append(keras.utils.to_categorical(1,num_classes))
                y.append(keras.utils.to_categorical(1,num_classes))
                y.append(keras.utils.to_categorical(1,num_classes))
                y.append(keras.utils.to_categorical(1,num_classes))
            if number >= 60 and number < 70:
                y.append(keras.utils.to_categorical(2,num_classes))
                y.append(keras.utils.to_categorical(2,num_classes))
                y.append(keras.utils.to_categorical(2,num_classes))
                y.append(keras.utils.to_categorical(2,num_classes))
            if number >= 70 and number < 80:
                y.append(keras.utils.to_categorical(2,num_classes))
                y.append(keras.utils.to_categorical(2,num_classes))
                y.append(keras.utils.to_categorical(2,num_classes))
                y.append(keras.utils.to_categorical(2,num_classes))
        return y   

#データの正規化など
def change_image(im):
    image_size = (32,32) #画像の学習サイズ
    #色空間を変換してリサイズ
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, image_size)
    # 配列の変形とデータの正規化
    im = im.reshape(32,32,3).astype('float32')/255

    return im


def makeDataByFlip(im):
    img2 = cv2.flip(im,1)
    h,w,colors = im.shape
    size = (w,h)
    # 切り捨て除算
    center = (w//2,h//2)

    angle = 5
    scale = 1.0
    matrix = cv2.getRotationMatrix2D(center,angle,scale)
    img2 = cv2.warpAffine(im,matrix,size)

    angle = 355
    matrix = cv2.getRotationMatrix2D(center,angle,scale)
    img3 = cv2.warpAffine(im,matrix,size)

    return img2,img3


#簡単な単体テスト
#path= os.getcwd()
#path_Hand = path + '/Data/HandDataImage'
#path_HandInfo = path + '/Data/HandInfo.csv'
#print(appendX(path_Hand))
#print(appendY(path_HandInfo))
#print(type(path))
#print(path + "/*.jpg")
#makeData = makeData()
#for i in range(500):
    #print(makeData.count_image())
#print(makeData.Data_20_imageNumber[1])

