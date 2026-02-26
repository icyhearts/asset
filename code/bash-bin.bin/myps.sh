#!/bin/bash
set -x
PID=$1
first=$(echo ${PID:0:1})
STRLEN=${#PID}
REMAIN_LEN=$((STRLEN-1))
remain=$(echo ${PID:1:${REMAIN_LEN}})
ps -eo pid,ppid,user,lstart,etime,cmd | grep --color "^ *[$first]${remain}"
