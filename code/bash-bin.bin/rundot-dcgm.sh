#!/bin/bash
set -x
cd /software_data/like/opt/nginx_install/html/dcgmNotes/images/
dot -Tsvg callGraph.dot -o ../output_images/callGraph.svg
echo $?
