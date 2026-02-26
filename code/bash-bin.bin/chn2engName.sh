#!/bin/bash
# function: rename any like "(a)" to "-a-"
#Wed, 11 Jan 2017 14:56:33 +0800
# set -x
WORK=$(pwd)
echo "Now working in $WORK"
find -maxdepth 1 | while read OLDFILENAME 
do 
	NEWFILENAME=$(echo $OLDFILENAME | sed 's/\ /_/g' | sed 's/(/_/g' |  sed 's/)/_/g' | sed 's/_\./\./g' | sed 's/__/_/g' | sed 's,通话录音,call_rec,g')
	if [ "$OLDFILENAME" != "$NEWFILENAME" ]; then
		echo "oldname="$OLDFILENAME 
		echo "newname="$NEWFILENAME
		mv "$OLDFILENAME"  "$NEWFILENAME"
	fi
done
