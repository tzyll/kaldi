#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.


import sys

'''
langs_to_filter format:
'Chinese English ...'

lre_matrix file format:
     Chinese English ...
utt1   0.2    0.6
utt2   0.5    0.3
...

'''
if len(sys.argv) != 3:
  print('usage: local/filter_lre_matrix.py langs_to_filter lre_matrix')
  sys.exit()

langs = sys.argv[1].strip().split()
langs_new = ''  # different order

ids = []
with open(sys.argv[2], 'r') as ls:
  lines = ls.readlines()
  # the 1st line is language names
  head = lines[0].strip().split()
  for i in range(len(head)):
    if head[i] in langs:
      ids.append(i)
      langs_new = langs_new + ' ' + head[i]
  print(langs_new)
  # the rest are scores
  for i in [line.strip().split() for line in lines[1:]]:
    new_l = i[0]
    for j in range(len(ids)):
      new_l = new_l + ' ' + i[ids[j]+1]
    print(new_l)
