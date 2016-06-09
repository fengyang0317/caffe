import cv2
import numpy as np
import sys
sys.path.insert(0, '../../python')
import caffe
import time

caffe.set_mode_cpu()

net = caffe.Net('deploy.prototxt', 'cars_nin_iter_2000.caffemodel', caffe.TEST)

mu = np.load('mean_val.npy')

im = cv2.imread('2012-12-12_12_30_08.jpg')
im -= mu
im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)

net.blobs['data'].reshape(1, 3, 360, 640)

net.blobs['data'].data[...] = im.transpose((2, 0 ,1))

st = time.time()
output = net.forward()
en = time.time()
print en - st

heatmap = output['pool3'][0]
heatmap /= heatmap.max()
heatmap = np.vstack((heatmap, np.zeros((1, heatmap.shape[1], heatmap.shape[2]), dtype=heatmap.dtype)))
heatmap = heatmap.transpose((1, 2, 0))

heatmap = cv2.resize(heatmap, (im.shape[1], im.shape[0]))

cv2.imshow('a', heatmap[:,:,0])
cv2.imshow('b', heatmap[:,:,1])
cv2.waitKey()