#!/bin/bash
set -x
cd /home/ice/tmp/speed/tokyo

start=`date +%s`
rm -rf vpsDir.tar.bz2*
wget http://vultr-tokyo/vpsDir.tar.bz2
end=`date +%s`
runtime=$((end-start))
MINS=$((runtime/60))
SECS=$((runtime%60))


echo "$(date -R) ${runtime}=${MINS} min ${SECS} sec"  >>/home/ice/tmp/wget-vultr-tokyo.txt
set +x
