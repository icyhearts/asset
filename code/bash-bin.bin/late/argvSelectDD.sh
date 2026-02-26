#!/bin/bash
set -x
for i in `seq $#`
do
	eval arg='$'$i
	if [[ -d "$arg" ]]; then
		echo "$arg is directory" 
		cd $arg
			find -type f | while read FNAME; do dd if=/dev/zero bs=1 count=1 of=$FNAME; done
		cd ..
	fi
done
