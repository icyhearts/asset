#!/bin/bash
set -x

cd /share_data/users/like/opt/nginx_install/html/bjh-html/sglangNotes/images
java -jar  ~/package//plantuml//plantuml.1.2023.6.jar   -tsvg class.pu -o ../output_images
echo $?
