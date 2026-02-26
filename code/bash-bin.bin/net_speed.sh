#!/bin/bash
set -x
LOG=/data0/shareVR/ping.txt
echo  "============================================================" >>${LOG}
/bin/date -R >>${LOG}
IPS=("mirrors.163.com" "baidu.com" "aliecs")
for IDX in ${!IPS[@]}; do
  ping -c 4 ${IPS[${IDX}]} >>${LOG} 2>&1
  echo  "-------------------------------------------------------------" >>${LOG}
done
set +x
