import os, cv2, lmdb, sys, random
import scipy.io as sio
import scipy.stats as stats
import numpy as np
sys.path.insert(0, '../../python')
import caffe

img_size = 260
map_size = 32

a = np.arange(1-map_size, map_size)
a = a*a
a.shape = 1,-1
b = a.transpose()
gau = a + b
gau = stats.norm.pdf(gau, 0, 1.5)
gau = gau * 4
#np.savetxt('g',gau)
#cv2.imshow('a', gau)
#cv2.waitKey(0)
#gau = gau/gau[63,63]

matdir = '/home/yfeng23/lab/pose/Release-v1.1/H36MDemo/val/'
id = 0
mean_val = np.zeros((3, img_size, img_size))
train = lmdb.open('lmdb_val_data', map_size=int(1e12))
label = lmdb.open('lmdb_val_label', map_size=int(1e12))
train_txn = train.begin(write=True)
label_txn = label.begin(write=True)
N = 200

lfile = os.listdir(matdir)
random.shuffle(lfile)
lvals = []
lrect = []
lmat = []
for fi in range(0, len(lfile), N):
    lframe = []
    lcap = [None] * fi
    for i in range(fi, fi+N):
        if i == len(lfile):
            break
        mat = sio.loadmat(matdir + lfile[i])
        lmat.append(mat)
        rect = mat['rect']
        pa = str(mat['Path'][0])
        na = str(mat['Name'][0])
        lcap.append(cv2.VideoCapture(pa + '/Videos/' + na +'.mp4'))
        lvals.append(mat['vals'])
        lrect.append(mat['rect'])
        for j in range(0, rect.shape[0]-2, 10):
            lframe.append((i, j))
    random.shuffle(lframe)

    for tu in lframe:
        print str(id) + ' ' + str(tu)
        cap = lcap[tu[0]]
        assert cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, tu[1])
        suc, frame = cap.read()
        assert suc
        if frame.shape[1] == 0:
            suc, frame = cap.read()
            assert suc
        assert frame.shape[0] > 999
        assert frame.shape[1] > 999
        assert frame.shape[2] == 3
        rec = lrect[tu[0]][tu[1]].astype(int)
        if rec[2] < rec[3]:
            rec[0] = rec[0] - (rec[3]-rec[2])/2
            rec[2] = rec[3]
            if rec[0] < 0:
                rec[0] = 0
            if rec[0] + rec[2] > frame.shape[1]:
                rec[0] = frame.shape[1] - rec[2]
        elif rec[2] > rec[3]:
            rec[1] = rec[1] - (rec[2]-rec[3])/2
            rec[3] = rec[2]
            if rec[1] < 0:
                rec[1] = 0
            if rec[1] + rec[3] > frame.shape[0]:
                rec[1] = frame.shape[0] - rec[3]
        frame=frame[rec[1]:rec[1]+rec[3],rec[0]:rec[0]+rec[2]]
        #cv2.imshow('a', frame)
        #cv2.waitKey(0)
        res = cv2.resize(frame, (img_size, img_size), interpolation = cv2.INTER_LINEAR)
        #cv2.imshow('a', res)
        #cv2.waitKey()
        res = res.transpose((2, 0, 1))
        mean_val = mean_val + res
        datum = caffe.io.array_to_datum(res)
        str_id = '{:08}'.format(id)
        id = id +1
        train_txn.put(str_id.encode('ascii'), datum.SerializeToString())
        part = lvals[tu[0]][tu[1]].copy()
        #print part.shape
        part[:,0] = part[:,0] - rec[0]
        part[:,1] = part[:,1] - rec[1]
        la = np.zeros((part.shape[0], map_size, map_size))
        for i in range(part.shape[0]):
            x = (map_size - 1) * part[i][0] / frame.shape[1]
            y = (map_size - 1) * part[i][1] / frame.shape[0]
            x = int(round(x))
            y = int(round(y))
            if x < 0 or y < 0 or x > map_size-1 or y > map_size-1:
                continue
            la[i] = gau[map_size-1-x:map_size*2-1-x,map_size-1-y:map_size*2-1-y]
            #cv2.imshow('b',la[i])
            #cv2.waitKey(0)
        datum = caffe.io.array_to_datum(la)
        label_txn.put(str_id.encode('ascii'), datum.SerializeToString())
        if id % 1000 == 0:
            train_txn.commit()
            label_txn.commit()
            print '----------processed ' + str(id) +' ----------'
            train_txn = train.begin(write=True)
            label_txn = label.begin(write=True)
train_txn.commit()
label_txn.commit()
train.close()
label.close()
mean_val = mean_val / len(lframe)
np.save('mean_val.npy', mean_val)
print str(id) + ' images in total'
