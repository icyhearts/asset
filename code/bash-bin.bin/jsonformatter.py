#!/usr/bin/python
import json
import datetime
import time
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-i', default='tmp/hbase-user-manager-set-coldbackup-request.json', type=str, help='year(4)-month(2)-date(2)')





global args
args = parser.parse_args()
path = args.i
newStr=json.dumps(json.loads(open(path).read()), indent=2)
open(path,'w').write(newStr)
