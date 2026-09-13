#set -x
sys_session_dir="/lib/x86_64-linux-gnu/.hist_sessions"
home_session_dir="/softhome/like/.bash_history_sessions/"
home_session_dir_docker="/softhome/like/.bash_history_sessions_docker/"
if [ -d "$sys_session_dir" ] && [ "$(stat -c '%U' "$sys_session_dir")" = "$USER" ]; then
  export HIST_DIR=$sys_session_dir
elif [ -d "$home_session_dir" ] && [ "$(stat -c '%U' "$home_session_dir")" = "$USER" ]; then
  export HIST_DIR=$home_session_dir
  mkdir -p ${HIST_DIR}
else
  export HIST_DIR=$home_session_dir_docker
  mkdir -p ${HIST_DIR}
fi
export HISTFILE=${HIST_DIR}/hist_$(hostname)

