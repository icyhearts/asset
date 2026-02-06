
# from claude code
# ============================================================
# 非交互式 SSH 命令历史记录
# 当 ssh host "cmd" 执行时，sshd 调用 bash -c "cmd"
# bash 会 source ~/.bashrc，利用这个时机记录命令
# ============================================================
if [[ $- != *i* ]] && [[ -n "$SSH_CONNECTION" ]] && [[ -n "$BASH_EXECUTION_STRING" ]]; then
    source ~/asset/code/hist_dir.sh
    {
        flock -x 200
        printf '#%s\n' "$(date +%s)" >> "$HISTFILE"
        printf '%s\n' "$BASH_EXECUTION_STRING" >> "$HISTFILE"
    } 200>"${HISTFILE}.lock"
fi
