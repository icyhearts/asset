#!/bin/bash
# CR LF --> LF
# set -x
if [[ "$#" -lt "1" ]]; then
	echo "usage: xxx.sh file1 file2 file3 ..."
	exit 2
fi
for IDX in `seq 1 $#`;
do
	eval filename='$'$IDX
	echo "dealing $filename"
	equery d $filename
done
