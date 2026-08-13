#!/usr/bin/env bash
while true
do
    # Your commands here
    bash  /softhome/like/bash-bin/bin/nvidia-smi-user.sh  /data/like/temp/smi.txt >> /data/like/temp/smi-loop.txt
    sleep 60
done

