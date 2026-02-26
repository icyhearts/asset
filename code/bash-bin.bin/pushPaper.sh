#!/bin/bash
cd /home/ice/shareVR/viewnetPaper/latexTemplate/Unix_LaTeX2e_Transactions_Style_File/IEEEtran
bash gitadd.sh
git commit -m "commited on `date +%Y-%m-%d___%H-%M-%S`"
git push pc master
