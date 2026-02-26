set -x

XB=/usr/bin/xtrabackup
XB_FILE_ROOT=/root/xtrabackup/tmp
DEFAULTS=/usr/local/mysql/mysql-5.7.26-linux-glibc2.12-x86_64/etc/my.cnf
USER=root
HOST=127.0.0.1
PORT=3001
LOG_FILE="/root/xtrabackup/logs/log.txt"

YEAR=`date +%Y`
MONTH=`date +%m`
DAY=`date +%d`


function nowstr () {
  date +%Y_%m_%d___%H_%M_%S
}

function loginfo() {
  echo "`nowstr`: $1" >> ${LOG_FILE}
}

XB_FILE_TODAY_DIR=${XB_FILE_ROOT}/${YEAR}/${MONTH}/${DAY}


XB_FILE_MONTH_DIR="${XB_FILE_ROOT}/${YEAR}/${MONTH}/"
mkdir -p ${XB_FILE_MONTH_DIR}
NEWEST_DAY_DIR=`ls -t ${XB_FILE_MONTH_DIR} | head -1`
if [[ "" == "${NEWEST_DAY_DIR}" ]]; then
  loginfo "month dir  empty, full backup, today as target dir"
  ${XB} --defaults-file=${DEFAULTS}  --backup --target-dir=${XB_FILE_TODAY_DIR} --user=${USER} --host=${HOST} --port=${PORT} >> ${LOG_FILE} 2>&1
else
  loginfo "month dir  not empty, inc backup, newest as base dir"
  ${XB} --defaults-file=${DEFAULTS}  --backup --target-dir=${XB_FILE_TODAY_DIR}  --incremental-basedir=${XB_FILE_MONTH_DIR}/${NEWEST_DAY_DIR}  --user=${USER} --host=${HOST} --port=${PORT}  >> ${LOG_FILE} 2>&1
fi
