import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def
img = cv.imread('/Users/tal.valani/PycharmProjects/final/test2.jpg')
if img is None:

mask = np.zeros(img.shape[:2], np.uint8)
print(img.shape)
height = int(img.shape[0] * 0.81)
width = int(img.shape[1] * 0.5)
left = int(img.shape[1] * 0.32)
top = int(img.shape[0] * 0.15)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (626, 240, 300, 320)
print(rect)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()