#!/bin/bash
set -x
for I in `ps aux | grep [l]ike12 | awk '{print $2}'`; do kill -9 $I; done
mv ~/.ssh/ssh_mux-* /tmp/
