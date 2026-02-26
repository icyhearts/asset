for pid in `ps aux | grep "[m]ake\.sh" | awk '{print $2}'`
do
	kill -9 $pid
done
for pid in `ps aux | grep "[s]impleMake.sh"   | awk '{print $2}'` 
do
	kill -9 $pid
done
