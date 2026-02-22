export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export EDITOR=vim
export PATH=/softhome/like/bash-bin/universal-ctags/bin/:/share_data/users/like/bash-bin/bin:/share_data/users/like/package/h100/package/cmake//github/cmake-3.26.0-rc5-linux-x86_64/bin/:$PATH
#export PS1="\D{%Y-%m-%d %H:%M:%S}|\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\u@\h:\w\$ "
export PS1="\h|\D{%Y-%m-%d %H:%M:%S}\[\e[01;34m\][\[\e[01;32m\]\u@ \W\[\e[01;34m\]]\[\e[00m\] "

export HF_DATASETS_CACHE=/share_data/users/like/huggingface_cache
export HF_ENDPOINT=https://hf-mirror.com

#cuda12_8=/share_data/users/like/opt/cuda-12.8/
#cuda13_0=/share_data/users/like/opt/cuda-13.0/

#hostname | grep  -E "gpu[0-9]{1,}.rd.sio-software.com"  && CUDA_HOME=$cuda12_8 ||  CUDA_HOME=$cuda13_0

set -o vi

# python3.13 no longer support vi mode, I need vi mode, PYTHON_BASIC_REPL=1 make me happy:https://github.com/python/cpython/issues/118840
export PYTHON_BASIC_REPL=1

## https://stackoverflow.com/questions/9457233/unlimited-bash-history
## Eternal bash history.
## ---------------------
## Undocumented feature which sets the size to "unlimited".
## http://stackoverflow.com/questions/9457233/unlimited-bash-history
#export HISTFILESIZE=
#export HISTSIZE=
#export HISTTIMEFORMAT="[%F %T] "
## Change the file location because certain bash sessions truncate .bash_history file upon close.
## http://superuser.com/questions/575479/bash-history-truncated-to-500-lines-on-each-login
#export HISTFILE=~/.bash_eternal_history
## Force prompt to write history after every command.
## http://superuser.com/questions/20900/bash-history-loss
#PROMPT_COMMAND="history -a; $PROMPT_COMMAND"
#

# stackoverflow version is not safe for multi termina and nfs
# claude code:
# ============================================================
# Bash Eternal History Configuration
# ============================================================

# 1. 历史文件位置
# export HISTFILE=~/.bash_eternal_history
# 确保目录存在
source ~/asset/code/hist_dir.sh
# 2. 历史记录无限制
export HISTSIZE=-1          # 内存中的历史条数（-1 = 无限）
export HISTFILESIZE=-1      # 历史文件的最大行数（-1 = 无限）

# 3. 历史格式：添加时间戳（可选但推荐）
export HISTTIMEFORMAT="%F %T  "

# 4. 忽略重复和空格开头的命令（可选）
export HISTCONTROL=ignoreboth

# 5. 多终端安全写入的核心配置
# -a: 立即追加当前会话的新命令到历史文件（不覆盖）
# -n: 从历史文件读取尚未读取的新行（其他终端写入的）
shopt -s histappend                    # 追加模式，不覆盖

# 6. 每次命令执行后立即写入历史文件
# PROMPT_COMMAND 在每次显示提示符前执行
# 使用 history -a 追加新命令，history -c 清空内存，history -r 重新加载
# 这样可以看到其他终端的命令（可选）

# 使用 flock 对历史文件加排他锁后再追加
__history_append_safe() {
    {
        flock -x 200
        history -a
    } 200>"${HISTFILE}.lock"
}
PROMPT_COMMAND="${PROMPT_COMMAND:+$PROMPT_COMMAND$'\n'}__history_append_safe"



# 7. 保存多行命令为单行（可选）
shopt -s cmdhist

# 8. 不限制历史文件大小（防止被系统截断）
# 某些系统可能有默认的 HISTSIZE 限制，这里再次确保
unset HISTSIZE HISTFILESIZE
export HISTSIZE=-1
export HISTFILESIZE=-1
