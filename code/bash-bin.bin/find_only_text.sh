#!/bin/bash
# https://stackoverflow.com/questions/4767396/linux-command-how-to-find-only-text-files
#find . -type f -exec grep -Il . {} +
find .  \( -path ./temp -o -path ./.git \) -prune -o -type f -exec grep -Il . {} +
