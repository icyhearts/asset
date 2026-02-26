#!/usr/bin/python3
import argparse
from cnocr import CnOcr
# import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--image', default='../examples/multi-line_cn1.png',  help='path')
global args
args = parser.parse_args()

ocr = CnOcr()
res = ocr.ocr(args.image)

print("Predicted Chars:")
for idx in range(len(res)):
	print(''.join(res[idx]))
