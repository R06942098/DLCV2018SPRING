
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:24:21 2018

@author: cengbowei
"""
#!/bin/bash 
wget 'https://www.dropbox.com/s/tcszv1q0z7bldm9/fc32.ckpt.data-00000-of-00001?dl=1' -O'fc32_vgg.ckpt.data-00000-of-00001'
wget 'https://www.dropbox.com/s/u9vol2bflwsx3gn/fc32.ckpt.index?dl=1' -O'fc32_vgg.ckpt.index'
wget 'https://www.dropbox.com/s/63efnpwpdltsqy7/fc32.ckpt.meta?dl=1' -O'fc32_vgg.ckpt.meta'
python3 hw3.py $1 $2 
