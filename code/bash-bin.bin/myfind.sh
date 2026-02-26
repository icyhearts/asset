#!/bin/bash
#for file in `find -maxdepth 1`
#do
#du -h $file | tail -1
#done

du -sh * .[a-zA-Z_-]* | sort -h
