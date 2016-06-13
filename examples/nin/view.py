import numpy as np
import cv2
import lmdb
import sys
sys.path.insert(0, '../../python')
import caffe

env = lmdb.open('lmdb_train', readonly=True)
mean_val = np.load('mean_val.npy')
with env.begin() as txn:
    '''
    for i in xrange(1000):
        raw_datum = txn.get('{:08}'.format(i).encode('ascii'))
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)

        flat_x = np.fromstring(datum.data, dtype=np.float)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        print datum.label
        cv2.imshow('a', x / 255)
        k = cv2.waitKey()
        if k == 27:
            break
    '''
    cursor = txn.cursor()
    for key, value in cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)

        x = caffe.io.datum_to_array(datum)
        print datum.label
        x = x.transpose((1,2,0))
        x += mean_val
        cv2.imshow('a', x / 255)
        k = cv2.waitKey()
        if k == 27:
            break
