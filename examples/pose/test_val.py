import caffe
import lmdb
import os
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
import numpy as np
import cv2

label_env = lmdb.open('lmdb_val_label')
data_env = lmdb.open('lmdb_val_data')
label_txn = label_env.begin()
data_txn = data_env.begin()
label_cursor = label_txn.cursor()
data_cursor = data_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()
used = [0, 2, 3, 7, 8, 14, 17, 18, 19, 25, 26, 27]

caffe.set_mode_gpu()

model_def = 'deploy.prototxt'
model_weights = 'pose4gait.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

sum = []

for key, value in label_cursor:
    datum.ParseFromString(value)
    label = caffe.io.datum_to_array(datum)
    label = label[used]

    data_cursor.next()
    datum.ParseFromString(data_cursor.value())
    data = caffe.io.datum_to_array(datum)
    if False:
        v = np.ones((32, 33 * 12))
        for i in range(12):
            v[:, i * 33:i * 33 + 32] = label[i]
        v = cv2.resize(v, None, fx=4, fy=4)
        cv2.imshow('label', v)
        cv2.imshow('data', data[0])
        cv2.waitKey()

    data -= 112

    net.blobs['data'].data[...] = data[0, 2:-2, 2:-2]
    output = net.forward()
    output_prob = output['conv8'][0]

    res = np.ones(12) * 64
    for i in xrange(12):
        heatmap = output_prob[i]
        pos = np.nonzero(heatmap == heatmap.max())
        gt = np.nonzero(label[i] == label[i].max())
        for x, y in zip(pos[0], pos[1]):
            dis = np.sqrt((x-gt[0][0])*(x-gt[0][0]) + (y-gt[1][0])*(y-gt[1][0]))
            res[i] = min(res[i], dis)
    sum.append(res)
np.save('distance', sum)