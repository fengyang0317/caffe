import sys
sys.path.insert(0, 'python')
import caffe
import cv2

caffe.set_mode_gpu()
solver = caffe.get_solver('models/VGGNet/VOC0712/SSD_300x300/solver.prototxt')

solver.net.forward()
im = solver.net.blobs['data'].data.transpose((0, 2, 3, 1))
for i in xrange(im.shape[0]):
    cv2.imshow('a', im[i] / 255)
    cv2.waitKey()