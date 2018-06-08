#!/usr/bin/env python3
# -*- coding: utf-8 -*-
wget 'https://www.dropbox.com/s/xigaoytj7ubp2yb/vgg16_weights.npz?dl=1' -O'vgg16_weights.npz'
wget 'https://www.dropbox.com/s/wk34urbtmkdw61g/150epochs.data-00000-of-00001?dl=1' -O'150epochs.data-00000-of-00001'
wget 'https://www.dropbox.com/s/thkzt32l3lpqisx/150epochs.index?dl=1' -O'150epochs.index'
wget 'https://www.dropbox.com/s/sbhcaqtw315nvos/150epochs.meta?dl=1' -O'150epochs.meta'
python3 CNN_features.py $1 $2 