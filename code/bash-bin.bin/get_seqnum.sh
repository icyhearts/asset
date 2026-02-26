#!/bin/bash
# set -x
# 得到等间距的数，以','为分隔符
if [[ "$#" -ne "3" ]]; then
	echo "usage: xxx.sh start skip stop"
    exit 2
fi
tmpf=`date +%Y_%m_%d___%H_%M_%S`
tmpf2="/tmp/${tmpf}.txt"

seq $1 $2 $3 >${tmpf2}
RESULT=`paste -s -d "," ${tmpf2}`
echo ${RESULT}
rm -rf ${tmpf2}
