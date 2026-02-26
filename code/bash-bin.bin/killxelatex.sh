#!/bin/bash
for i in `ps aux | grep [x]elatex   | awk '{print $2}'`; do echo $i; kill -9 $i; done
