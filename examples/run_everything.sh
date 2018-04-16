#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

for script in all_methods.py imagenet_lrp.py ; do
#for net in vgg16 vgg19 resnet50 inception_v3 inception_resnet_v2 densenet121 densenet169 densenet201 nasnet_large nasnet_mobile ; do
for net in nasnet_mobile nasnet_large ; do
    logfile=./logs/$net-$script.txt
    echo running $script. writing output to $logfile
    python3 $script $net >> $logfile
    echo ""
done
done
echo "run_everything.sh has terminated."
