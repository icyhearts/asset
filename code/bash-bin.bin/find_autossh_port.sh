#!/bin/bash
set -x
ps aux | grep [a]utossh | awk -F"-M" '{print $2}' | awk '{print $1}' | sort -n
