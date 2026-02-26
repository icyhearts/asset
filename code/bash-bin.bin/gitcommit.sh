#!/bin/bash
BRANCH=$(git branch | grep "\*" | awk '{print $2}')
git commit -m "[`date +%Y/%m/%d___%H:%M:%S`*${BRANCH}]$1"
