#!/bin/bash
# function: rename any like "(a)" to "-a-"
#Wed, 11 Jan 2017 14:56:33 +0800
# set -x
WORK=$(pwd)
echo "Now working in $WORK"
# read的是带./的文件名和目录名: ./a_b_c
# ./高清_1080P_日语课程_标日初级精讲BY萌萌哒葉子先生_叶子老师完整课程及后续中高级课程请看详情介绍_P2_第二课_这是书.flv
find -maxdepth 1 | while read OLDFILENAME_DOT
do 
  #首先把 开头的 "./" 去掉: ./foo/abc.txt -> foo/abc.txt
  OLDFILENAME=$(echo $OLDFILENAME_DOT | sed 's,^\./,,g')
  #太长了，先把OLDFILENAME转换成NEWFILENAME1:
  #                               |英文问号     |多个空格           |[1,inf)个英文左圆括号|[1,inf)个英文右圆括号|[1,inf)个_接[1,inf)个.     |[1,inf)个[         |[1,inf)个]            
  NEWFILENAME1=$(echo $OLDFILENAME|sed 's/?/_/g'|sed 's/ \{1,\}/_/g'| sed 's/(\{1,\}/_/g' | sed 's/)\{1,\}/_/g' |sed 's/_\{1,\}\.\{1,\}/./g'|sed 's/\[\{1,\}/_/g'| sed 's/]\{1,\}/_/g')
  #再把NEWFILENAME1转换成NEWFILENAME:
  #                               |[1,inf)个中文方圆括号问号冒号   ||[1,inf)个英文,    |开头的[1,inf)个_   |[1,inf)个_          
#   NEWFILENAME=$(echo $NEWFILENAME1|sed 's/[】【？]\{1,\}/_/g' | sed 's/（\{1,\}/_/g'| sed 's/）\{1,\}/_/g'|sed 's/,\{1,\}/_/g'|sed 's/^_\{1,\}//g'|sed 's/_\{1,\}/_/g')
  # 合并所有中文的标点符号
  NEWFILENAME=$(echo $NEWFILENAME1|sed 's/[（）：】【？]\{1,\}/_/g'|sed 's/,\{1,\}/_/g'|sed 's/^_\{1,\}//g'|sed 's/_\{1,\}/_/g')
  if [ "$OLDFILENAME" != "$NEWFILENAME" ]; then
      echo "oldname="$OLDFILENAME 
      echo "newname="$NEWFILENAME
      mv "$OLDFILENAME"  "$NEWFILENAME"
  fi
done
