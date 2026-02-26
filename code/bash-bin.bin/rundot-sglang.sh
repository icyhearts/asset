#!/bin/bash
set -x
cd /share_data/users/like/opt/nginx_install/html/bjh-html/sglangNotes/images
dot -Tsvg callGraph.dot -o ../output_images/callGraph.svg
echo $?
