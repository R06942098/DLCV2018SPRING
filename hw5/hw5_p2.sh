#!/usr/bin/env python3
# -*- coding: utf-8 -*-
wget 'https://www.dropbox.com/s/xigaoytj7ubp2yb/vgg16_weights.npz?dl=1' -O'vgg16_weights.npz'
wget 'https://www.dropbox.com/s/5h98qg2vivj022g/5.data-00000-of-00001?dl=1' -O'5.data-00000-of-00001'
wget 'https://www.dropbox.com/s/6dpll1z7ua0880l/5.index?dl=1' -O'5.index'
wget 'https://www.dropbox.com/s/k0muyjm4cvcfwvo/5.meta?dl=1' -O'5.meta'
python3 RNN_features.py $1 $2