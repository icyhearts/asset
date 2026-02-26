#!/bin/bash
# 把tab换成4个space，常用于python文件中
set -x
if [[ "$#" -ne "1" ]]; then
	echo "usage: xxx.sh fileName"
fi
sed -i 's/^    /\t/g' $1
sed -i 's/^\t    /\t\t/g' $1
sed -i 's/^\t\t    /\t\t\t/g' $1
sed -i 's/^\t\t\t    /\t\t\t\t/g' $1
sed -i 's/^\t\t\t\t    /\t\t\t\t\t/g' $1
sed -i 's/^\t\t\t\t\t    /\t\t\t\t\t\t/g' $1
sed -i 's/^\t\t\t\t\t\t    /\t\t\t\t\t\t\t/g' $1
