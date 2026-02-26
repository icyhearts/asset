#!/bin/bash
# provide url, wget -c it on server, tar on server, down tar to local disk; so time is saved
set -x
usage() {
echo "usage: xx.sh -d subdir_to_save -u url_to_wget -p port_to_ssh_port"
}
DOWNHOST=vultr-tokyo2
HTTP_ROOT=/var/www/html/
MID_VIDEO_DIR=video/
FINAL_STR=""
URL=""
while getopts "d:u:p:" OPTION; do
    case $OPTION in
    d)
        FINAL_STR=$OPTARG
        echo "OPTIND="$OPTIND
        ;;
    u)
        URL=$OPTARG
        echo "OPTIND="$OPTIND
        ;;
    p)
        PORT=$OPTARG
        echo "PORT="$PORT
        ;;
    *)
        usage
        exit 1
        ;;
    esac
done
echo "OPTIND="$OPTIND
echo "FINAL_STR=${FINAL_STR}"
echo "URL=${URL}"
if [[ -z "$FINAL_STR" ]]; then
    echo "you forget to provide sub dir name"
    usage
    exit 1
fi

if [[ -z "$URL" ]]; then
    echo "you forget to provide sub url"
    usage
    exit 1
fi

if [[ -z "$PORT" ]]; then
    echo "you forget to provide port number"
    usage
    exit 1
fi

VIDEODIR=${HTTP_ROOT}/${MID_VIDEO_DIR}/${FINAL_STR}
# 
CMD_FILE_NAME="$(date +%Y_%m_%d___%H_%M_%S).sh"
CMD_FILE_PATH="/tmp/${CMD_FILE_NAME}"
echo -e "set -x\nwget -O ${FINAL_STR}.mp4 -c \"${URL}\" >/dev/null 2>&1 ; echo ret=\$?" >${CMD_FILE_PATH}
ssh -p ${PORT} root@${DOWNHOST} "mkdir -p ${VIDEODIR}" || exit 1
scp -P ${PORT} ${CMD_FILE_PATH} root@${DOWNHOST}:$VIDEODIR || exit 1
ssh -p $PORT root@$DOWNHOST "cd ${VIDEODIR} && bash ${CMD_FILE_NAME}" || exit 1
echo ">wget -c done"
# 
ZIPPED_FILE_NAME=${FINAL_STR}.tar.bz2
ssh -p $PORT root@${DOWNHOST} "cd ${HTTP_ROOT}/${MID_VIDEO_DIR} && tar -jcf ${ZIPPED_FILE_NAME} ${FINAL_STR} && md5sum ${ZIPPED_FILE_NAME}"  || exit 1
echo ">tar done"
axel -n 10 -a http://$DOWNHOST/${MID_VIDEO_DIR}/${ZIPPED_FILE_NAME} || exit 1
md5sum ${ZIPPED_FILE_NAME}
echo ">down from vps done"
# 
set +x
