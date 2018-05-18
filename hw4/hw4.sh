#!/bin/bash 
wget 'https://www.dropbox.com/s/sn1bt2qpz21nc0n/140epochs.data-00000-of-00001?dl=' -O'140epochs.data-00000-of-00001'
wget 'https://www.dropbox.com/s/oz4s4gyukbbw2bp/140epochs.index?dl=1' -O'140epochs.index'
wget 'https://www.dropbox.com/s/2kujumzu9qc6cu3/140epochs.meta?dl=1' -O'140epochs.meta'
wget 'https://www.dropbox.com/s/1rp0olwfy5htqsc/ac_110epochs.data-00000-of-00001?dl=1' -O'ac_110epochs.data-00000-of-00001'
wget 'https://www.dropbox.com/s/r6frk7ksfbxonaw/ac_110epochs.index?dl=1' -O'ac_110epochs.index'
wget 'https://www.dropbox.com/s/kzuxve5rb9e8nuq/ac_110epochs.meta?dl=1' -O'ac_110epochs.meta'
wget 'https://www.dropbox.com/s/b427fxiszwlv6gl/ac_smile.pickle?dl=1' -O'ac_smile.pickle'
wget 'https://www.dropbox.com/s/rzymuc22k7bvvgi/dc_no_major.pickle?dl=1' -O'dc_no_major.pickle'
wget 'https://www.dropbox.com/s/kzzw0ihogp31841/info_ma.data-00000-of-00001?dl=1' -O'info_ma.data-00000-of-00001'
wget 'https://www.dropbox.com/s/uhvak2dn72i6xhh/info_ma.index?dl=1' -O'info_ma.index'
wget 'https://www.dropbox.com/s/zyq8xclxievdv5v/info_ma.meta?dl=1' -O'info_ma.meta'
wget 'https://www.dropbox.com/s/3p18bqtw5ix4qbl/info_long_short.pickle?dl=1' -O'info_long_short.pickle'
wget 'https://www.dropbox.com/s/oaqgjzg8hvsik5e/vae.data-00000-of-00001?dl=1' -O'vae.data-00000-of-00001'
wget 'https://www.dropbox.com/s/etirqtcf98v6zod/vae.index?dl=1' -O'vae.index'
wget 'https://www.dropbox.com/s/zupke4t55usa5nw/vae.meta?dl=1' -O'vae.meta'
wget 'https://www.dropbox.com/s/y2j55wk4uk6j8sn/vae_no.pickle?dl=1' -O'vae_no.pickle'
python3 hw4_problem1.py $1 $2 
python3 hw4_problem2.py $1 $2 
python3 hw4_problem3.py $1 $2 
python3 hw4_bonus.py $1 $2 

