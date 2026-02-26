#!/usr/bin/python
import sys
import re

filename = sys.argv[1]
with  open(filename) as ifp:
    lines = ifp.readlines()

k_node = "N[0-9]+"
k_page_size_kb = "kernelpagesize_kB"

other_keys = ["active", "anon", "dirty", "mapmax", "mapped"]

node_pages = dict()
page_size_found = False
page_size_kb = 4
for line in lines:
    line = line.strip()
    fields = line.split()
    for field in fields:
        mat_node = re.match('{}='.format(k_node), field)
        if mat_node:
            node=field[0:mat_node.end()-1]
            pages=eval(field[mat_node.end():])
            if node not in node_pages:
                node_pages[node] = []
            node_pages[node].append(pages)

        if not page_size_found:
            mat = re.match('{}='.format(k_page_size_kb), field)
            if mat:
                page_size_kb = eval(field[mat.end():])
                page_size_found = True
                print("page_size_found:{}".format(page_size_kb))


        for key  in other_keys:
            mat = re.match('{}='.format(key), field)
            if mat:
                k=field[0:mat.end() - 1]
                v=eval(field[mat.end():])
                if k not in node_pages:
                    node_pages[k] = []
                node_pages[k].append(v)


for k in sorted(node_pages.keys()):
    num_pages = sum(node_pages[k])
    num_gb = num_pages * page_size_kb / 1024**2
    print("{:15}: {:15} pages ({:.3f} GiB)".format(k, num_pages, num_gb))
