#!/bin/bash
set -x
VIM_GIT_DIR="/home/ice/shareVR/vimConfig"

cd $VIM_GIT_DIR

cp .vim/after/ftplugin/c.vim  .vim/after/ftplugin/vim.vim .vim/after/ftplugin/sh.vim .vim/after/ftplugin/python.vim .vim/after/ftplugin/make.vim ~/.vim/after/ftplugin/
cp .vimrc ~/.vimrc
