set -x
alias u7ssh="ssh -oPubkeyAcceptedKeyTypes=+ssh-dss -oHostKeyAlgorithms=+ssh-dss  -oKexAlgorithms=diffie-hellman-group1-sha1"
alias u7scp="scp -oPubkeyAcceptedKeyTypes=+ssh-dss -oHostKeyAlgorithms=+ssh-dss  -oKexAlgorithms=diffie-hellman-group1-sha1"
export GIT_SSH_COMMAND="ssh -oPubkeyAcceptedKeyTypes=+ssh-dss -oHostKeyAlgorithms=+ssh-dss  -oKexAlgorithms=diffie-hellman-group1-sha1"
set +x
