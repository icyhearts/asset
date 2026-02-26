#!/bin/bash
set -x
if [[ "$#" -ne "2" ]]; then
	echo "usage: xx.sh input_fname output_fname"
fi
# first get number of line 
NL=`wc -l  $1  | awk '{print $1}'`
# then use tac to reverse and redirect
tail -$NL $1 | tac >$2
echo $NL
