#!/bin/bash
set -x
cd /mast/shareVR/notes/notes_dev_repo/call_graph
dot -Tsvg onnxmnist.dot   -O
echo $?

