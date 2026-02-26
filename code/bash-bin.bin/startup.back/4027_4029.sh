#!/bin/bash
/usr/bin/autossh -o StrictHostKeyChecking=no -M 4027 -p28337 -fCNR 4029:localhost:80 root@vpsgfw &&  echo "yes, `date`" >>/home/ice/date4027.txt      || echo "no, `date`" >>/home/ice/date4027.txt
