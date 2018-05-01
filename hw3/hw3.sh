
@author: cengbowei
#!/bin/bash 

wget 'https://www.dropbox.com/s/tcszv1q0z7bldm9/fc32.ckpt.data-00000-of-00001?dl=1' -O'fc32_vgg.ckpt.data-00000-of-00001'
wget 'https://www.dropbox.com/s/u9vol2bflwsx3gn/fc32.ckpt.index?dl=1' -O'fc32_vgg.ckpt.index'
wget 'https://www.dropbox.com/s/63efnpwpdltsqy7/fc32.ckpt.meta?dl=1' -O'fc32_vgg.ckpt.meta'
python hw3.py $1 $2 
