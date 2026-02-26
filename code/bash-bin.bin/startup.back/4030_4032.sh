#!/bin/bash
# 
/usr/bin/autossh -o StrictHostKeyChecking=no -M 4030 -p22 -fCNR 4032:localhost:22303 keli@smile10k &&  echo "yes, `date`" >>/mast/start_log/date4030.txt      || echo "no, `date`" >>/mast/start_log/date4030.txt
