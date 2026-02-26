#!/bin/bash
set -x
if [[ "$#" -ne "1" ]]; then
	echo "usage: xx.sh dir_prefix "
	exit 1
else
	mkdir "$1_`date +%Y_%m_%d__%H_%M_%S`" 
fi
