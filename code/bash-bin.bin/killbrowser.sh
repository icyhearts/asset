#!/bin/bash
#for i in `ps aux | grep [c]hrome   | awk '{print $2}'`; do echo $i; kill -9 $i; done
#for i in `ps aux | grep [c]hromium   | awk '{print $2}'`; do echo $i; kill -9 $i; done
for i in `ps aux | grep -E "([c]hromium|[c]hrome|[f]irefox)"   | awk '{print $2}'`; do echo $i; kill -9 $i; done
