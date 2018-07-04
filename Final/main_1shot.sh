"""
Created on Thu Jan 18 13:24:21 2018
@author: cengbowei
"""


#!/bin/bash 
wget 'https://www.dropbox.com/s/sxtsquob6hvyp56/59.data-00000-of-00001?dl=1' -O'59.data-00000-of-00001'
wget 'https://www.dropbox.com/s/7d4o63iic634dvp/59.index?dl=1' -O'59.index'
wget 'https://www.dropbox.com/s/b3lsdixs9wag9hy/59.meta?dl=1' -O'59.meta'
wget 'https://www.dropbox.com/s/bontq6lppbiaxb2/sample_1_shot.npy?dl=1' -O'sample_1_shot.npy'
python3 main_1.py --novel_path $1 --test_path $2 --outfile $3 --test True