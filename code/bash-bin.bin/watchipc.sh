set -x
PAT=/tmp/ipc*.txt
DELAY=3

do_notify() {
  for IPCFILE in $(ls $PAT); do
    INFO=$(head -1 $IPCFILE)
    ~/bin/mynotify.sh $INFO
    rm $IPCFILE
  done
}

while true; do
  ls $PAT && do_notify
  sleep $DELAY
done
