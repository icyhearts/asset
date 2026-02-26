#!/bin/bash
# 删除文件，只保留其中的1/3
set -x
find -type f | awk '{if(NR%9!=0) print $1}'   | xargs rm
