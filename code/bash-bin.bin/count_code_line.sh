#!/bin/bash
# tell you how many lines are there in specific suffix
find -type f -regextype egrep -regex '.*\.(h|hpp|c|cc|cpp)' | xargs wc -l
