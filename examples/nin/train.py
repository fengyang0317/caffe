import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2

def nin(lmdb, batch_size):
    n = caffe.NetSpec()
    c1_kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='gaussian', std=0.05),
        'bias_filler': dict(type='constant')
    }
    c2_kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        'weight_filler': dict(type='gaussian', std=0.05),
        'bias_filler': dict(type='constant', value=0)
    }
    c3_kwargs = {
        'param': [dict(lr_mult=0.1, decay_mult=1), dict(lr_mult=0.1, decay_mult=0)],
        'weight_filler': dict(type='gaussian', std=0.05),
        'bias_filler': dict(type='constant', value=0)
    }

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=192, pad=2, **c1_kwargs)
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.cccp1 = L.Convolution(n.relu1, kernel_size=1, num_output=160, group=1, **c2_kwargs)
    n.relu_cccp1 = L.ReLU(n.cccp1, in_place=True)
    n.cccp2 = L.Convolution(n.relu_cccp1, kernel_size=1, num_output=96, group=1, **c2_kwargs)
    n.relu_cccp2 = L.ReLU(n.cccp2, in_place=True)
    n.pool1 = L.Pooling(n.relu_cccp2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.drop3 = L.Dropout(n.pool1, in_place=True, dropout_ratio=0.5)

    n.conv2 = L.Convolution(n.drop3, kernel_size=5, num_output=192, pad=2, **c1_kwargs)
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.cccp3 = L.Convolution(n.relu2, kernel_size=1, num_output=192, group=1, **c2_kwargs)
    n.relu_cccp3 = L.ReLU(n.cccp3, in_place=True)
    n.cccp4 = L.Convolution(n.relu_cccp3, kernel_size=1, num_output=192, group=1, **c2_kwargs)
    n.relu_cccp4 = L.ReLU(n.cccp4, in_place=True)
    n.pool2 = L.Pooling(n.relu_cccp4, kernel_size=3, stride=2, pool=P.Pooling.AVE)
    n.drop6 = L.Dropout(n.pool2, in_place=True, dropout_ratio=0.5)

    n.conv3 = L.Convolution(n.drop6, kernel_size=3, num_output=192, pad=1, **c1_kwargs)
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.cccp5 = L.Convolution(n.relu3, kernel_size=1, num_output=192, group=1, **c2_kwargs)
    n.relu_cccp5 = L.ReLU(n.cccp5, in_place=True)
    n.cccp6 = L.Convolution(n.relu_cccp5, kernel_size=1, num_output=2, group=1, **c3_kwargs)
    n.relu_cccp6 = L.ReLU(n.cccp6, in_place=True)
    n.pool3 = L.Pooling(n.relu_cccp6, kernel_w=8, kernel_h=16, stride=1, pool=P.Pooling.AVE)
    n.accuracy = L.Accuracy(n.pool3, n.label, include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    n.loss = L.SoftmaxWithLoss(n.pool3, n.label)

    return n.to_proto()

with open('auto_train.prototxt', 'w') as f:
    f.write('name: "CIFAR10_full"\n')
    f.write(str(nin('lmdb_train', 128)))