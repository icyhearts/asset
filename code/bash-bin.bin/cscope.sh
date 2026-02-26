#!/bin/bash
# https://codeyarns.com/tech/2017-08-24-how-to-install-and-use-cscope.html#gsc.tab=0
find -iregex '.*\.\(h\|cc\|hpp\|c\|cpp\|cu\)$' > .files_for_cscope
cscope -i .files_for_cscope -b

