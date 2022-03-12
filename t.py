import os
import glob
import sys
import numpy as np
import os.path as osp
import random
import csv
from PIL import Image


line = 0
with open("id_list.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        line += 1
        if line < 4:
            continue
        for i in range(5,11,1):
            if row[i] == '1':
               print(f"tag[{line - 3}] = {i-4}") 
               input()
