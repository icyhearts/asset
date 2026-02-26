#!/bin/bash
set -x
PID=$1

pstree -pa | grep --color "[,]$1" -A 20 -B 20
