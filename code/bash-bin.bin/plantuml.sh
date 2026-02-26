#!/bin/bash
set -x
cd /home/like12/localhost/notes_dev_repo/pu
java -jar ~/package/plantuml/plantuml.1.2023.6.jar  -tsvg unipredict.pu
echo $?
