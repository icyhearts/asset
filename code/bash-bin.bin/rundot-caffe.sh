#!/bin/bash
set -x
cd /var/www/localhost/htdocs/notes_pc_repo/caffeNotes/images
dot -Tsvg caffeCallGraph.dot -o ../output_images/caffeCallGraph.svg
echo $?
