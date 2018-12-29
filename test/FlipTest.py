
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import os, glob
import matplotlib.pyplot as plt

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


path= os.getcwd()
path_Hand = path + '/Data/smallHandDataImage'
files = glob.glob(path_Hand + "/*.jpg")


for f in files:
    print("ok")
    im = cv2.imread(f)
    im2,im3 = makeDataByFlip(im)
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(im3,cv2.COLOR_BGR2RGB))
    plt.show()

