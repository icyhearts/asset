#!/usr/bin/python3
import re
import argparse

## argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--ifname',  default='aabb.txt', help='filename you want to deal')
parser.add_argument('--ofname',  default='aabb.txt', help='filename you want to output')
global args
args = parser.parse_args()
## end of argparse
ifname = args.ifname
with open(ifname) as ifp:
  lines = ifp.readlines()
pat = re.compile(r'^(    ){1,}(.+)')
new_lines = []
for lno,line  in enumerate(lines, 1):
  mat = re.search(pat, line)
  if mat:
    #找到group 2的开始索引，索引是以0为基的
    first_non_4space_idx = mat.start(2)
    new_line = ' '* int(first_non_4space_idx/2) + line[first_non_4space_idx:]
  else:
    new_line = line
  new_lines.append(new_line)
full_text = ''.join(new_lines)
with open(args.ofname, 'w') as ofp:
  ofp.write(full_text )
print('done')
