#!/bin/bash
set -x
grep -E "(new.*$CLASS|make_.*$CLASS|$CLASS *[({])" -n -r $@
