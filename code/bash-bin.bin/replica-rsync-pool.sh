rsync -acvP /home/ice/shareVR /pool/replica/home/ice
rsync -acvP --exclude="*YouCompleteMe" /home/ice/.vim /pool/replica/home/ice
rsync -acvP /home/ice/.vimrc  /pool/replica/home/ice
rsync -acvP /home/ice/bin /pool/replica/home/ice
rsync -acvP /etc/conf.d/docker  /pool/replica/etc/conf.d
