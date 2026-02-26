#!/usr/bin/python3
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-i', default=224, type=int, help='input size')
parser.add_argument('-p', default=0, type=int, help='padding')
parser.add_argument('-d', default=1, type=int, help='dilation')
parser.add_argument('-k', default=3, type=int, help='kernel size')
parser.add_argument('-s', default=1, type=int, help='stride ')



def main():
	global args
	args = parser.parse_args()
	i=args.i

	p=args.p
	d=args.d
	k=args.k
	s=args.s

	out=np.floor( 1.*(i+2*p- d*(k-1)-1)/s + 1.)
	print('Conv2d and MaxPool2d i={},out={}'.format(i,out))
	# for avgpool
	out=np.floor( 1.*(i+2*p- k)/s + 1.)
	print('AvgPool2d, i={},out={}'.format(i,out))

if __name__ == '__main__':
	main()

print('end')
