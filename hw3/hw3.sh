
@author: cengbowei
#!/bin/bash 
wget 'https://www.dropbox.com/s/fm312m3wquv9igv/fc32.ckpt.data-00000-of-00001?dl=1' -O'fc32_vgg.ckpt.data-00000-of-00001'
wget 'https://www.dropbox.com/s/2sx52528hktuntt/fc32.ckpt.index?dl=1' -O'fc32_vgg.ckpt.index'
wget 'https://www.dropbox.com/s/3aya3mcld0lbs3v/fc32.ckpt.meta?dl=1' -O'fc32_vgg.ckpt.meta'
python hw3.py $1 $2

