#!/bin/bash
# 
#Tue, 14 Mar 2017 13:42:22 +0800
# @function: scp commandline argument files to a host
# @bug: filename don't have space or tab. i.e. 'to me.jpp' is not allowed, 
IP="192.168.40.157"
TRANSFER_FILES=""
for i in `seq 1 $#`
do
	eval arg='$'$i
	TRANSFER_FILES=$TRANSFER_FILES' '$arg
done
echo "TRANSFER_FILES=$TRANSFER_FILES"
#scp -rp $TRANSFER_FILES root@$IP:/mnt/gentoo/laptop/
scp -rp $TRANSFER_FILES ice@$IP:/home/ice/dataVR/
