#!/bin/bash
# for i in `ps aux | grep [c]hrome   | awk '{print $2}'`; do echo $i; kill -9 $i; done
set -x
for i in `ps aux | grep [F]oxit | grep -i "\.[p]df"  | awk '{print $2}'`
	do echo $i
	kill -9 $i
done
