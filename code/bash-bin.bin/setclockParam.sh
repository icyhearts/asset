#!/bin/bash
set -x
echo "notify-send -u critical \"$2\"" |  at now + $1 minutes 
