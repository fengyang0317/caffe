import numpy as np
import sys
import cv2
cv2.imshow('a', np.zeros((10, 10)))
sys.path.insert(0, 'python')
import caffe
import os
import time

if not os.path.exists('output'):
    os.mkdir('output')

caffe.set_mode_gpu()
net = caffe.Net('models/ResNet/rail/SSD_200x200/deploy_.prototxt',
                'models/ResNet/rail/SSD_200x200/rail_SSD_200x200_iter_60000.caffemodel',
                caffe.TEST)

mu = np.array([110, 108, 101])

imdir = '/home/de566945/yang/rail/imgs'
imgs = os.listdir(imdir)

#net.blobs['data'].reshape(1, 3, 180, 320)

tsum = 0
n = 0

#for i in open('/home/yfeng23/intern/prepare/list.txt', 'r'):
for i in imgs:
    #i = i.split(' ')[0]
    na, ext = os.path.splitext(i)
    #if i.startswith('pos'):
    #    continue
    #i = i[4:]
    if ext != '.png' and ext != '.jpg':
        continue
    im = cv2.imread(os.path.join(imdir, i))
    #im = cv2.imread('/home/yfeng23/intern/ssd/examples/ssd/2012-12-12_12_30_08_r.jpg')
    #im = cv2.imread('/home/yfeng23/intern/ssd/examples/ssd/dump_output_018359.jpg')
    width = im.shape[1]
    height = im.shape[0]
    net.blobs['data'].reshape(1, 3, height, width)
    tmp = im.astype(float) - mu
    net.blobs['data'].data[...] = tmp.transpose((2, 0, 1))
    st = time.time()
    output = net.forward()
    en = time.time()
    tsum += en - st
    bbox = output['detection_out'][0,0]
    for j in xrange(bbox.shape[0]):
        if bbox[j, 2] > 0.1:
            cv2.rectangle(im,(int(bbox[j, 3] * width), int(bbox[j, 4] * height)),
                              (int(bbox[j, 5] * width), int(bbox[j, 6] * height)), (255, 0, 0))
    cv2.imshow('a', im)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join('output', i), im)
    n += 1
print tsum, n
print tsum / n