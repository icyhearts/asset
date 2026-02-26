#!/bin/bash
# set -x
FNAME="$(date +%Y_%m_%d).txt"
DSTR=$(date +%Y_%m_%d___%H_%M_%S)
echo $DSTR >>  /pool/logs/ntpdate/${FNAME}
ntpdate pool.ntp.org >>  /pool/logs/ntpdate/${FNAME}
