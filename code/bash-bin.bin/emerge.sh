#!/bin/bash
set -x
emerge --sync 2>&1 | tee /tmp/emerge.log
