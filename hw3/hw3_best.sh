@author: cengbowei
#!/bin/bash 
wget 'https://www.dropbox.com/s/unbetnya2kfvlzf/fc16_vgg.ckpt.data-00000-of-00001?dl=1' -O'fc16_vgg.ckpt.data-00000-of-00001'
wget 'https://www.dropbox.com/s/une9q1ftlmw6eeq/fc16_vgg.ckpt.index?dl=1' -O'fc16_vgg.ckpt.index'
wget 'https://www.dropbox.com/s/unfmzf2h9s3qtfr/fc16_vgg.ckpt.meta?dl=1' -O'fc16_vgg.ckpt.meta'
python hw3_best.py $1 $2