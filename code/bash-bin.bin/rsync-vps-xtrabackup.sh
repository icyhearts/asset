mkdir -p /mast/logs/rsync-vps-xtrabackup/
rsync -acP -e "ssh -p 28337" root@vpsgfw:/root/xtrabackup /mast/vps_xtrabackup > /mast/logs/rsync-vps-xtrabackup/`date +%Y_%m_%d___%H_%M_%S`.log 2>&1
