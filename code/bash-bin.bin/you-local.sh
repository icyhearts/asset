#!/bin/bash
# provide url, you-get it on server, tar on server, down tar to local disk; so time is saved
set -x
usage() {
echo "usage: xx.sh -d subdir_to_save -u url_to_you-get -p port_to_ssh_port"
}
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
    usage
    exit 1
fi

if [[ -z "$URL" ]]; then
    usage
    exit 1
fi

if [[ -z "$PORT" ]]; then
    usage
    exit 1
fi

VIDEODIR=$HTTP_ROOT/${MID_VIDEO_DIR}/${FINAL_STR}
# 
CMD="you-get $URL"
# 
DOWNHOST=static.likesite.win
ssh -p $PORT root@$DOWNHOST "mkdir -p $VIDEODIR && cd ${VIDEODIR} && ${CMD}" || exit 1
echo ">you-get done"
# 
MD5FILE=md5.${FINAL_STR}.txt
ssh -p $PORT root@$DOWNHOST "cd $HTTP_ROOT/$MID_VIDEO_DIR && tar -cf ${FINAL_STR}.tar ${FINAL_STR} && md5sum ${FINAL_STR}.tar >${MD5FILE}"  || exit 1
echo ">tar done"
#wget -c http://$DOWNHOST/${MID_VIDEO_DIR}/${FINAL_STR}.tar || exit 1
#axel -n 10 -a https://$DOWNHOST/${MID_VIDEO_DIR}/${FINAL_STR}.tar || exit 1
wget -c https://$DOWNHOST/${MID_VIDEO_DIR}/${FINAL_STR}.tar || exit 1
#axel -n 10 -a https://$DOWNHOST/${MID_VIDEO_DIR}/${MD5FILE} || exit 1
wget -c https://$DOWNHOST/${MID_VIDEO_DIR}/${MD5FILE} || exit 1
md5sum -c ${MD5FILE} || exit 1
echo ">down from vps done"
# 
set +x
