#!/bin/bash
set -x
find -type f | while read FNAME; do dd if=/dev/zero bs=1 count=1 of=$FNAME; done
