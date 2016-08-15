import numpy as np
import sys
sys.path.insert(0, 'python')
import caffe
import os
import cv2
import time

if not os.path.exists('output'):
    os.mkdir('output')

caffe.set_mode_gpu()
net = caffe.Net('models/ResNet/cars/SSD_140x100/deploy.prototxt',
                'models/ResNet/cars/SSD_140x100/comp_cars_SSD_140x100_iter_10000.caffemodel',
                caffe.TEST)

mu = np.array([119, 121, 124])

imdir = '/home/yfeng23/lab/luxury_cars/data'
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
        if bbox[j, 2] > 0.5:
            cv2.rectangle(im,(int(bbox[j, 3] * width), int(bbox[j, 4] * height)),
                              (int(bbox[j, 5] * width), int(bbox[j, 6] * height)), (255, 0, 0))
    cv2.imshow('a', im)
    cv2.waitKey()
    cv2.imwrite(os.path.join('output', i), im)
    n += 1
print tsum, n
print tsum / n