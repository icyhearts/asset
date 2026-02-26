#!/bin/bash
set -x
cd /share_data/users/like/opt/nginx_install/html/bjh-html/vllmNotes/images
dot -Tsvg callGraph.dot -o ../output_images/callGraph.svg
echo $?
