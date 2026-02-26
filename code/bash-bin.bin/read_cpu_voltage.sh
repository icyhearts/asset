#!/bin/bash
# set -x
# https://askubuntu.com/questions/876286/how-to-monitor-the-vcore-voltage
watch -n1 'echo "scale=2; $(rdmsr 0x198 -u --bitfield 47:32)/8192" | bc'

