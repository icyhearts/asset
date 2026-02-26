#!/bin/bash
if [ "$#" -lt "1" ]; then
	echo "too few argument, use:$0 file_to_be_dealt"
	exit 1
fi
sed -i '/^$/d' $1
