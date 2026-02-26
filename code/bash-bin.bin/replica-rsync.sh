rsync -acvP /home/ice/shareVR /mast/replica/home/ice
rsync -acvP --exclude="*YouCompleteMe" /home/ice/.vim /mast/replica/home/ice
rsync -acvP /home/ice/.vimrc  /mast/replica/home/ice
rsync -acvP /home/ice/bin /mast/replica/home/ice
rsync -acvP /etc/conf.d/docker  /mast/replica/etc/conf.d
