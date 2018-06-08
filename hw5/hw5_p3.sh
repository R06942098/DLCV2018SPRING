#!/usr/bin/env python3
# -*- coding: utf-8 -*-
wget 'https://www.dropbox.com/s/xigaoytj7ubp2yb/vgg16_weights.npz?dl=1' -O'vgg16_weights.npz'
wget 'https://www.dropbox.com/s/k4i7csoi8b2gs41/model_1.data-00000-of-00001?dl=1' -O'model_1.data-00000-of-00001'
wget 'https://www.dropbox.com/s/z8ykxb4e7ow4xwr/model_1.index?dl=1' -O'model_1.index'
wget 'https://www.dropbox.com/s/p33rv4kck31fmfi/model_1.meta?dl=0' -O'model_1.meta'
python3 Beamsearch_SUB.py $1 $2