import os, cv2, lmdb, sys, random, h5py
import scipy.io as sio
import scipy.stats as stats
import numpy as np
sys.path.insert(0, '../../python')
import caffe


def makerect(rec, shape):
    lrec = []
    rec[0] = max(0, rec[0])
    rec[1] = max(0, rec[1])
    rec[2] = min(rec[2], shape[1] - rec[0])
    rec[3] = min(rec[3], shape[0] - rec[1])
    if rec[2] == rec[3]:
        lrec.append(rec)
        return lrec
    r1 = rec
    r2 = rec
    if rec[2] < rec[3]:
        r1[3] = r1[2]
        r2[1] += r2[3] - r2[2]
        r2[3] = r2[2]
    elif rec[2] > rec[3]:
        r1[2] = r1[3]
        r2[0] += r2[2] - r2[3]
        r2[2] = r2[3]
    lrec.append(r1)
    lrec.append(r2)
    return lrec

dbdir = '/home/yfeng23/lab/dataset/mpii_human_pose_v1_u12_2/'

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

val_data = lmdb.open('lmdb_val_data', map_size=int(1e12))
val_label = lmdb.open('lmdb_val_label', map_size=int(1e12))
train_data = lmdb.open('lmdb_train_data', map_size=int(1e12))
train_label = lmdb.open('lmdb_train_label', map_size=int(1e12))
tid = 0
vid = 0
tdata_txn = train_data.begin(write=True)
tlabel_txn = train_label.begin(write=True)
vdata_txn = val_data.begin(write=True)
vlabel_txn = val_label.begin(write=True)

mat = sio.loadmat(dbdir + 'mpii_human_pose_v1_u12_1.mat')
re = mat['RELEASE']
anno = re['annolist'][0][0].squeeze()
img_train = re['img_train'][0][0].squeeze()
for i in xrange(img_train.size):
    ro = anno[i]
    rect = ro['annorect']
    if rect.dtype == 'uint8' or rect.dtype.names.count('annopoints') == 0\
            or rect.dtype.names.count('scale') == 0 or rect.dtype.names.count('objpos') == 0:
        continue
    name = str(ro['image'][0][0][0][0])
    print i, name
    img = cv2.imread(dbdir + 'images/' + name)
    print img.shape
    for j in xrange(rect.size):
        scale = rect[0, j]['scale'][0][0] * 100
        px = rect[0, j]['objpos']['x'][0][0][0][0]
        py = rect[0, j]['objpos']['y'][0][0][0][0]
        pos = rect[0, j]['annopoints'][0][0][0]
        for k in xrange(pos.size):
            if pos['is_visible'][0, k].size == 0 or pos['is_visible'][0, k][0][0] == '0' \
                    or pos['is_visible'][0, k][0][0] == 0:
                continue
            lx = int(pos['x'][0, k][0][0])
            ly = int(pos['y'][0, k][0][0])
            scale = max(scale, 1.1 * abs(lx - px))
            scale = max(scale, 1.1 * abs(ly - py))
        scale = int(scale)
        print px, py, scale
        # cv2.circle(img, (px, py), 5, (0, 0, 255))
        rec = np.array([px-scale, py-scale, 2*scale, 2*scale])
        boxes = makerect(rec, img.shape[0:2])
        for box in boxes:
            res = img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
            cv2.imshow('a', res)
            cv2.waitKey()
            res = cv2.resize(res, (img_size, img_size))
            res = res.transpose((2, 0, 1))
            datum = caffe.io.array_to_datum(res)
            if img_train[j] == 1:
                tid += 1
                str_id = '{:08}'.format(tid)
                tdata_txn.put(str_id.encode('ascii'), datum.SerializeToString())
            else:
                vid += 1
                str_id = '{:08}'.format(vid)
                vdata_txn.put(str_id.encode('ascii'), datum.SerializeToString())
            la = np.zeros((16, map_size, map_size))
            for k in xrange(pos.size):
                if pos['is_visible'][0, k].size == 0 or pos['is_visible'][0, k][0][0] == '0' \
                        or pos['is_visible'][0, k][0][0] == 0:
                    continue
                lx = int(pos['x'][0, k][0][0])
                ly = int(pos['y'][0, k][0][0])
                if lx >= box[0] and lx <= box[0]+box[2] and ly >= box[1] and ly <= box[1]+box[3]:
                    jid = pos['id'][0, k][0][0]
                    lx -= box[0]
                    ly -= box[1]
                    lx = lx * (map_size - 1) / box[2]
                    ly = ly * (map_size - 1) / box[3]
                    lx = int(round(lx))
                    ly = int(round(ly))
                    la[jid] += gau[map_size-1-lx:map_size*2-1-lx,map_size-1-ly:map_size*2-1-ly]
            for k in xrange(la.shape[0]):
                fuse = res.transpose((1,2,0))
                fuse = cv2.cvtColor(fuse, cv2.COLOR_BGR2GRAY)
                fuse = fuse/255.0 + cv2.resize(la[k], (img_size, img_size))
                #cv2.imshow('a', fuse)
                #cv2.imshow('b', la[k])
                #cv2.waitKey()
            datum = caffe.io.array_to_datum(la)
            if img_train[j] == 1:
                tlabel_txn.put(str_id.encode('ascii'), datum.SerializeToString())
            else:
                vlabel_txn.put(str_id.encode('ascii'), datum.SerializeToString())
        if tid % 1000 == 0:
            tdata_txn.commit()
            tlabel_txn.commit()
            print '----------processed train ' + str(tid) +' ----------'
            tdata_txn = train_data.begin(write=True)
            tlabel_txn = train_label.begin(write=True)
        if vid % 1000 == 0:
            vdata_txn.commit()
            vlabel_txn.commit()
            print '----------processed val ' + str(tid) +' ----------'
            vdata_txn = val_data.begin(write=True)
            vlabel_txn = val_label.begin(write=True)
'''
        rec[1] = min(img.shape[0], rec[1])
        rec[3] = min(img.shape[1], rec[3])
        cv2.rectangle(img, (px-scale, py-scale), (px+scale, py+scale), (255, 0, 0), 5)
        #cv2.imshow('a',img[rec[0]:rec[1],rec[2]:rec[3]])
        for k in xrange(pos.size):
            if pos['is_visible'][0, k].size == 0 or pos['is_visible'][0, k][0][0] == '0' \
                    or pos['is_visible'][0, k][0][0] == 0:
                continue
            lx = int(pos['x'][0, k][0][0])
            ly = int(pos['y'][0, k][0][0])
            print lx, ly
            cv2.circle(img, (lx, ly), 5, (0, 255, 255))
        cv2.imshow('b', img)
        cv2.waitKey()
'''
tdata_txn.commit()
tlabel_txn.commit()
train_data.close()
train_label.close()
vdata_txn.commit()
vlabel_txn.commit()
val_data.close()
val_label.close()
print tid, ' images in total', vid