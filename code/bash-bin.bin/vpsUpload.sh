#!/bin/bash
set -x
cd /var/www/localhost/vps/
bash gitadd.sh
BRANCH=$(git branch | grep "\*" | awk '{print $2}')
git commit -m "commited on [`date +%Y-%m-%d___%H-%M-%S`*${BRANCH}]"
git push vps master
