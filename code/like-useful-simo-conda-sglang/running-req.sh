FILE=$1
echo "$FILE prefill"
grep "Prefill batch,.*#running-req" $FILE  | awk -F, '{print $6}'  | awk '{print $2}' | sort -h|  tail -3

echo "$FILE decode"
grep "Decode batch.*#running-req" $FILE | awk -F',' '{print $2}' | awk '{print $2}' | sort -h | tail -3
