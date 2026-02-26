#!/bin/bash
/usr/bin/autossh -o StrictHostKeyChecking=no -M 4024 -p28337 -fCNR 4026:localhost:22303 root@97.64.21.155 &&  echo "yes, `date`" >>/home/ice/date4024.txt      || echo "no, `date`" >>/home/ice/date4024.txt
