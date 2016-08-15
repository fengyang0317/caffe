import numpy as np
import sys
sys.path.insert(0, 'python')
import caffe
import os
import cv2
import time

caffe.set_mode_gpu()
net = caffe.Net('models/ResNet/face0/SSD_300x300/deploy.prototxt',
                'models/ResNet/face0/SSD_300x300/face_SSD_300x300_iter_60000.caffemodel',
#net = caffe.Net('models/ResNet/face/SSD_300x300/deploy.prototxt',
#                'models/ResNet/face/SSD_300x300/face_SSD_300x300_iter_30000.caffemodel',
                caffe.TEST)

mu = np.array([106, 113, 123])

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print frame.shape
width = frame.shape[1]
height = frame.shape[0]
net.blobs['data'].reshape(1, 3, height, width)
#cv2.imwrite('a.png', frame)

while(True):
    ret, frame = cap.read()
    if not ret:
        break

    tmp = frame.astype(float) - mu
    net.blobs['data'].data[...] = tmp.transpose((2, 0, 1))
    st = time.time()
    output = net.forward()
    en = time.time()
    cv2.putText(frame, 'fps={}'.format(int(1/(en-st))), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    bbox = output['detection_out'][0,0]
    for j in xrange(bbox.shape[0]):
        if bbox[j, 2] > 0.5:
            cv2.rectangle(frame, (int(bbox[j, 3] * width), int(bbox[j, 4] * height)),
                              (int(bbox[j, 5] * width), int(bbox[j, 6] * height)), (0, 255, 0), 2)
    frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
    cv2.imshow('a', frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()