TEMP=$(getopt -o "u:l:r:x:h" --long "user:,ld:,rd:,xavier:,help" -- "$@")
if [[ "$?" != "0" ]] ; then echo "getopt error, Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"

LOCAL_DIR=binaries
REMOTE_DIR=/vblkdev3/users/dongxu/perception_onboard/
XAVIER=xavier-b
X_USER=nvidia
help() {
cat << EOF
usage: env.sh OPTIONS
-i, --inet  interface
  set interface to use
-h, --help
  display this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help ) help; exit 0;;
    -l | --ld ) LOCAL_DIR=$2; shift 2;;
    -r | --rd ) REMOTE_DIR="$2"; shift 2 ;;
    -x | --xavier ) XAVIER="$2"; shift 2 ;;
    -u | --user ) X_USER="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done


echo "LOCAL_DIR=$LOCAL_DIR"
echo "REMOTE_DIR=$REMOTE_DIR"
echo "XAVIER=$XAVIER"
echo "X_USER=$X_USER"
set -x


git log -n 10 --oneline >$LOCAL_DIR/gitlog.txt
git diff >$LOCAL_DIR/diff.patch

rm -rf $LOCAL_DIR

bash tools/build_app_test_package.sh $LOCAL_DIR
TAR=$LOCAL_DIR.tar
tar -cf $TAR $LOCAL_DIR
md5sum $TAR >md5.txt

scp $TAR  md5.txt $X_USER@$XAVIER:$REMOTE_DIR/
ssh  $X_USER@$XAVIER "cd $REMOTE_DIR/; md5sum -c md5.txt"

set +x
