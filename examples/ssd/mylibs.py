import caffe
from caffe import layers as L
from caffe import params as P

c1_kwargs = {
    'param': [dict(lr_mult=1, decay_mult=1)],
    'weight_filler': dict(type='xavier'),
    'bias_term': False,
}


def wide_basic(net, from_layer, prefix, nInputPlane, nOutputPlane, stride):
    #assert from_layer in net.keys()

    ln_branch2a_name = '{}_ln_branch2a'.format(prefix)
    net[ln_branch2a_name] = L.LayerNorm(net[from_layer], in_place=True, eps=0.001)
    relu_branch2a_name = '{}_relu_branch2a'.format(prefix)
    net[relu_branch2a_name] = L.ReLU(net[ln_branch2a_name], in_place=True)
    branch2a_name = '{}_branch2a'.format(prefix)
    net[branch2a_name] = L.Convolution(net[relu_branch2a_name], num_output=nOutputPlane, kernel_size=3, pad=1,
                                       stride=stride, **c1_kwargs)

    ln_branch2b_name = '{}_ln_branch2b'.format(prefix)
    net[ln_branch2b_name] = L.LayerNorm(net[branch2a_name], in_place=True, eps=0.001)
    relu_branch2b_name = '{}_relu_branch2b'.format(prefix)
    net[relu_branch2b_name] = L.ReLU(net[ln_branch2b_name], in_place=True)
    branch2b_name = '{}_branch2b'.format(prefix)
    net[branch2b_name] = L.Convolution(net[relu_branch2b_name], num_output=nOutputPlane, kernel_size=3, pad=1,
                                       stride=1, **c1_kwargs)
    if nInputPlane != nOutputPlane:
        branch1_name = '{}_branch1'.format(prefix)
        net[branch1_name] = L.Convolution(net[relu_branch2a_name], num_output=nOutputPlane, kernel_size=1,
                                          stride=stride, pad=0, **c1_kwargs)
        net[prefix] = L.Eltwise(net[branch2b_name], net[branch1_name])
    else:
        net[prefix] = L.Eltwise(net[branch2b_name], net[from_layer])


def layer(net, from_layer, prefix, nInputPlane, nOutputPlane, count, stride):
    l_name = '{}_0'.format(prefix)
    wide_basic(net, from_layer, l_name, nInputPlane, nOutputPlane, stride)
    for i in xrange(1, count):
        c_name = '{}_{}'.format(prefix, i)
        wide_basic(net, l_name, c_name, nInputPlane, nOutputPlane, 1)
        l_name = c_name


def WideResNetBody(net, from_layer, depth = 10, widen_factor = 1):
    assert from_layer in net.keys()

    assert (depth - 4) % 6 ==0, '%6=0'
    n = (depth - 4) / 6
    nStages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor,
               128 * widen_factor, 256 * widen_factor, 512 * widen_factor]
    net['conv1'] = L.Convolution(net[from_layer], num_output=nStages[0], kernel_size=3, stride=1, pad=1, **c1_kwargs)
    layer(net, 'conv1', 'res2a', nStages[0], nStages[1], n, 1)
    layer(net, 'res2a_{}'.format(n-1), 'res3a', nStages[1], nStages[2], n, 2)
    layer(net, 'res3a_{}'.format(n-1), 'res4a', nStages[2], nStages[3], n, 2)
    layer(net, 'res4a_{}'.format(n-1), 'res5a', nStages[3], nStages[4], n, 2)
    layer(net, 'res5a_{}'.format(n-1), 'res6a', nStages[4], nStages[5], n, 2)
    #layer(net, 'res6a_{}'.format(n-1), 'res7a', nStages[5], nStages[6], n, 2)
    net['ln'] = L.LayerNorm(net['res6a_{}'.format(n-1)], in_place=True, eps=0.001)
    net['relu'] = L.ReLU(net['ln'], in_place=True)

    '''
    net.ip6 = L.Convolution(net['relu'], num_output=64, kernel_size=3, pad=1, stride=1, **c1_kwargs)
    net.ip6_bn = L.BatchNorm(net.ip6, in_place=True, **bn_kwargs)
    net.ip6_relu = L.ReLU(net.ip6_bn, in_place=True)
    net.ip7 = L.Convolution(net.ip6_relu, num_output=64, kernel_size=1, pad=0, stride=1, **c1_kwargs)
    net.ip7_bn = L.BatchNorm(net.ip7, in_place=True, **bn_kwargs)
    net.ip7_relu = L.ReLU(net.ip7_bn, in_place=True)
    '''
    return net.to_proto()