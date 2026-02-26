#!/bin/bash
set -x
cd /var/www/localhost/htdocs/notes_pc_repo/jemallocNotes/images/
dot -Tsvg jemallocCallGraph.dot -o ../output_images/jemallocCallGraph.svg
echo $?
