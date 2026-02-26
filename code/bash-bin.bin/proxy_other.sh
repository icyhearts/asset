#!/bin/bash
/usr/bin/autossh -o StrictHostKeyChecking=no -M 4012 -p28337 -fCNR 4014:localhost:2080 root@47.111.241.61
/usr/bin/autossh -o StrictHostKeyChecking=no -M 4015 -p28337 -fCNR 4017:localhost:2081 root@47.111.241.61
/usr/bin/autossh -o StrictHostKeyChecking=no -M 4021 -p28337 -fCNR 4023:localhost:7001 root@47.111.241.61
