import os, cv2, lmdb, sys, random, h5py
import scipy.io as sio
import scipy.stats as stats
import numpy as np
sys.path.insert(0, '../../python')
import caffe

isColor = True
isVal = False

def makerect(rec, shape):
    if rec[2] < rec[3]:
        rec[0] = rec[0] - (rec[3]-rec[2])/2
        rec[2] = rec[3]
        if rec[0] < 0:
            rec[0] = 0
        if rec[0] + rec[2] > shape[1]:
            rec[0] = shape[1] - rec[2]
    elif rec[2] > rec[3]:
        rec[1] = rec[1] - (rec[2]-rec[3])/2
        rec[3] = rec[2]
        if rec[1] < 0:
            rec[1] = 0
        if rec[1] + rec[3] > shape[0]:
            rec[1] = shape[0] - rec[3]
    return rec

gaitdir = '/home/yfeng23/lab/dataset/cmu_mobo/moboJpg/'
def getbg():
    people = os.listdir(gaitdir)
    pid = np.random.randint(len(people))
    angles = os.listdir(gaitdir + people[pid] + '/bgImage')
    aid = np.random.randint(len(angles))
    bgfile = os.listdir(gaitdir + people[pid] + '/bgImage/' + angles[aid])
    bg = cv2.imread(gaitdir + people[pid] + '/bgImage/' + angles[aid] + '/' + bgfile)
    return bg

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

if isVal:
    matdir = '/home/yfeng23/lab/pose/Release-v1.1/H36MDemo/val/'
    train = lmdb.open('lmdb_val_data', map_size=int(1e12))
    label = lmdb.open('lmdb_val_label', map_size=int(1e12))
else:
    matdir = '/home/yfeng23/lab/pose/Release-v1.1/H36MDemo/data/'
    train = lmdb.open('lmdb_train_data', map_size=int(1e12))
    label = lmdb.open('lmdb_train_label', map_size=int(1e12))
id = 0
if isColor:
    mean_val = np.zeros((3, img_size, img_size))
else:
    mean_val = np.zeros((img_size, img_size))
train_txn = train.begin(write=True)
label_txn = label.begin(write=True)
N = 200

lfile = os.listdir(matdir)
random.shuffle(lfile)
lvals = []
lrect = []
for fi in range(0, len(lfile), N):
    lframe = []
    lcap = [None] * fi
    for i in range(fi, fi+N):
        if i == len(lfile):
            break
        f = h5py.File(matdir + lfile[i])
        rect = np.array(f['rect']).transpose()
        pa = str(''.join(map(unichr, f['Path'][:])))
        na = str(''.join(map(unichr, f['Name'][:])))
        lcap.append(cv2.VideoCapture(pa + '/Videos/' + na +'.mp4'))
        lvals.append(np.array(f['vals']).transpose())
        lrect.append(rect)
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
        if ~isColor:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        f = h5py.File(matdir + lfile[tu[0]])
        d = f['bgmask']
        e = f[d[tu[1]][0]]
        bg = np.array(e)
        bg = bg.transpose()
        bg = bg[rec[1]:rec[1]+rec[3],rec[0]:rec[0]+rec[2]]
        #cv2.imshow('a', frame)
        #cv2.waitKey(0)
        res = cv2.resize(frame, (img_size, img_size), interpolation = cv2.INTER_LINEAR)
        bgmask = cv2.resize(bg, (img_size, img_size), interpolation = cv2.INTER_LINEAR)
        bg = getbg()
        bg = cv2.resize(bg, (img_size, img_size), interpolation = cv2.INTER_LINEAR)
        res[bgmask==0] = bg[bgmask==0]
        #print np.mean(res[bgmask!=0])
        #cv2.imshow('a', res)
        #cv2.waitKey()
        if ~isColor:
            res.shape = res.shape[0], res.shape[1], 1
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
            y = (map_size - 1) * part[i][0] / frame.shape[1]
            x = (map_size - 1) * part[i][1] / frame.shape[0]
            x = int(round(x))
            y = int(round(y))
            if x < 0 or y < 0 or x > map_size-1 or y > map_size-1:
                continue
            la[i] = gau[map_size-1-x:map_size*2-1-x,map_size-1-y:map_size*2-1-y]
            hm = cv2.resize(la[i], (img_size, img_size));
            fu = res/255.0 + hm
            #cv2.imshow('fuse', fu.transpose((1,2,0)))
            #cv2.waitKey(0)
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
np.save('mean_train.npy', mean_val)
print str(id) + ' images in total'
