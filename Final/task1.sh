#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:24:21 2018
@author: cengbowei
"""


#!/bin/bash 
wget 'https://www.dropbox.com/s/rpeq4prv9w8u2n5/99.data-00000-of-00001?dl=1' -O'99.data-00000-of-00001'
wget 'https://www.dropbox.com/s/6i0ddv70ugimmxl/99.index?dl=1' -O'99.index'
wget 'https://www.dropbox.com/s/9897j4oz308k5wi/99.meta?dl=1' -O'99.meta'
python3 main.py --test_path $1 --outfile $2 --test True 