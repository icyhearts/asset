#!/usr/bin/python3
import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-s1', default=0.0, type=float, help='area 1')
parser.add_argument('-p1', default=0.0, type=float, help='price 1')

parser.add_argument('-s2', default=0.0, type=float, help='area 2')
parser.add_argument('-p2', default=0.0, type=float, help='price 2')


def main():
    global args
    args = parser.parse_args()
    sold_area = args.s2 - args.s1
    sold_total_price = args.s2 * args.p2 - args.s1 * args.p1
    sold_average = sold_total_price / sold_area
    print(f"sold area:{sold_area}, sold average price:{sold_average}")


if __name__ == '__main__':
    main()
