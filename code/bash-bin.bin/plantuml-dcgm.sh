#!/bin/bash
set -x

cd /software_data/like/opt/nginx_install/html/dcgmNotes/images
java -jar  /software_data/like/package/plantuml//plantuml.1.2023.6.jar   -tsvg class.pu -o ../output_images
echo $?
