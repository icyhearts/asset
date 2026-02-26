#!/bin/bash
/usr/bin/autossh -o StrictHostKeyChecking=no -M 4009 -p28337 -fCNR 4011:localhost:7002 root@boostup.cf
