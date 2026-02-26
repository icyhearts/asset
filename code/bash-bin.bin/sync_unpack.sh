TEMP=$(getopt -o "h" --long "user:,host:,file:,rd:,unpack:,help" -- "$@")
if [[ "$?" != "0" ]] ; then echo "getopt error, Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"

USER=hadoop
HOST=ice-qemu
FILE=a.tar
RD=/mast/opt
UNPACK="tar -xf "
help() {
cat << EOF
usage: env.sh OPTIONS
-i, --inet  interface
  set interface to use
-h, --help
  display this message
EOF
}

set -x
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help ) help; exit 0;;
    --user ) USER=$2; shift 2;;
    --host ) HOST="$2"; shift 2 ;;
    --file ) FILE="$2"; shift 2 ;;
    --rd ) RD="$2"; shift 2 ;;
    --unpack ) UNPACK="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

ssh $USER@$HOST "mkdir -p $RD"
rsync -acvP $FILE  $USER@$HOST:$RD
BASE_FILENAME=$(basename $FILE)
ssh $USER@$HOST "cd $RD; $UNPACK $BASE_FILENAME"
set +x
