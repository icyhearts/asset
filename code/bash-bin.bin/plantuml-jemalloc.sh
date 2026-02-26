#!/bin/bash
set -x
cd /var/www/localhost/htdocs/notes_pc_repo/jemallocNotes/images
java -jar /mast/packageLinux/plantuml/plantuml.1.2023.6.jar  -tsvg jemallocClass.pu -o ../output_images
echo $?
