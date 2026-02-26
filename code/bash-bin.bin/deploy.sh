set -e
TEMP=$(getopt -o "u:l:r:x:h" --long "pack-unzip-dir:,instance-dir:,local-conf-dir:,remote-conf-dir:,deploy-hosts:,deploy-user:,re-unzip:,help" -- "$@")
if [[ "$?" != "0" ]] ; then echo "getopt error, Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"

function remove_trail_slash(){
  if [[ "$#" == "0" ]]; then
  echo ""
  fi
  IN=$1
  OUT="$(dirname $IN)/$(basename $IN)"
  echo $OUT
}
PACK_UNZIP_DIR=hbase-1.3.0
INSTANCE_DIR=/mast/opt/hadoop/instances/instance3/
LOCAL_CONF_DIR=""
REMOTE_CONF_DIR=""
DEPLOY_HOSTS="ice-qemu,ice-qemu-node1,ice-qemu-node2"
DEPLOY_USER=hadoop
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
    --pack-unzip-dir ) PACK_UNZIP_DIR=$2; shift 2;;
    --instance-dir ) INSTANCE_DIR=$2; shift 2;;
    --local-conf-dir ) LOCAL_CONF_DIR=$2; shift 2;;
    --remote-conf-dir ) REMOTE_CONF_DIR=$2; shift 2;;
    --deploy-hosts ) DEPLOY_HOSTS=$2; shift 2;;
    --deploy-user ) DEPLOY_USER=$2; shift 2;;
    ##
    -- ) shift; break ;;
    * ) break ;;
  esac
done

#FUNC_UNZIP_DIR=$(remove_trail_slash $PACK_UNZIP_DIR)
#PACK_UNZIP_DIR=$(echo ${PACK_UNZIP_DIR} | sed "s,/$,,g")
PACK_UNZIP_DIR=$(remove_trail_slash $PACK_UNZIP_DIR)

IFS=', ' read -r -a HOST_ARRAY <<< "$DEPLOY_HOSTS"
# 1: unzip to local
for HOST in ${HOST_ARRAY[@]}; do
  echo  "$HOST"
  # 2: rsync from local to remote and delte conf
  ssh -t ${DEPLOY_USER}@${HOST} "mkdir -p ${INSTANCE_DIR}"
  rsync -acP --delete ${PACK_UNZIP_DIR}  ${DEPLOY_USER}@${HOST}:${INSTANCE_DIR}
done

if [[ -n "$LOCAL_CONF_DIR" ]] && [[ -n "$REMOTE_CONF_DIR" ]]; then
  LOCAL_CONF_DIR=$(remove_trail_slash $LOCAL_CONF_DIR)
  REMOTE_CONF_DIR=$(remove_trail_slash $REMOTE_CONF_DIR)
  echo "sync $LOCAL_CONF_DIR to $REMOTE_CONF_DIR"
  for HOST in ${HOST_ARRAY[@]}; do
    rsync -acP --delete ${LOCAL_CONF_DIR}  ${DEPLOY_USER}@${HOST}:${REMOTE_CONF_DIR}
  done
fi

set +x

