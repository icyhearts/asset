#!/bin/bash
set -x
for i in `seq $#`
do
	eval arg='$'$i
#	echo "arg="$arg
	if [[ -d "$arg" ]]; then
		echo "$arg is directory" 
		cd $arg
			find -type f | awk '{if(NR%5!=0) print $1}'   | xargs rm
		cd ..
	fi
done
