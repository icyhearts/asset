#set -x
sys_session_dir="/lib/x86_64-linux-gnu/.hist_sessions"
if [ -d "$sys_session_dir" ] && [ "$(stat -c '%U' "$sys_session_dir")" = "$USER" ]; then
  export HIST_DIR=$sys_session_dir
else
  export HIST_DIR=~/.bash_history_sessions
  mkdir -p ${HIST_DIR}
fi
export HISTFILE=${HIST_DIR}/hist_$(hostname)

