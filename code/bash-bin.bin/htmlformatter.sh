#!/bin/bash
# 处理 html文件
# 替换html中的'<' 为 &lt;
# 替换html中的'>' 为 &gt;
if [[ "$#" -eq "2" ]]; then
	set -x
	sed -i "${1} s/</\&lt;/g" ${2}
	sed -i "${1} s/>/\&gt;/g" ${2}
else
	set +x
	echo "# 处理 html文件"
	echo "# 替换html中的'<' 为 &lt;"
	echo "# 替换html中的'>' 为 &gt;"
	echo "xx.sh line_specify filename. line_specify can be sed format like: 2,4 or 1 or 1,4 or 2 etc"
fi
