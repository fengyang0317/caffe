import os
import random
import sys
import itertools

random.seed(0)

dp = '/home/yfeng23/study/vision/grading/Taolue/testfile/'

l = list()

for g in ['MALE', 'FEMALE']:
    imgs = os.listdir(dp + g)
    imgs.remove('.directory')
    if g == 'MALE':
        gi = '0'
    else:
        gi = '1'
    for i in imgs:
        l.append(dp + g + '/' + i + ' ' + gi)

random.shuffle(l)
part = 0

if len(sys.argv) > 1:
    part = int(sys.argv[1])

n = len(l)
s = [(i * 10 / n) == part for i in xrange(n)]
ns = [(i * 10 / n) != part for i in xrange(n)]

test = list(itertools.compress(l, s))
train = list(itertools.compress(l, ns))

with open('test.txt', 'w') as f:
    for i in test:
        f.write('%s\n' % i)

with open('train.txt', 'w') as f:
    for i in train:
        f.write('%s\n' % i)