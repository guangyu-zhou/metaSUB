from __future__ import print_function

import numpy as np
import pandas as pd
# import gmplot
import pickle
from gmplot import GoogleMapPlotter as gmp
from geopy.distance import vincenty
from scipy import spatial
from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt
import pylab
from itertools import combinations
def load_RNA():
    with open("../ncbi/16SMicrobial_trim.fasta", 'r') as rna_db:
        dict = {}
        key = ''
        # cnt = 1000
        for line in rna_db:
            line = line.rstrip('\n')
            if '>' in line:
                dict[line] = ''
                key = line
                continue

            dict[key]+=line

            # cnt-=1
            # if cnt < 0:
                # break
    print(len(dict.keys()))
    return dict
    # for elem in dict:
    #     print(elem, len(dict[elem]))



# rna_db = load_RNA()
# all_species_df = np.loadtxt("../files/species_names_tree.txt", dtype='str', delimiter = '\n')
# all_species_df = np.loadtxt("../files/bacteria_species.txt", dtype='str', delimiter = '\n')
all_species_df = np.loadtxt("../files/16sRNA/missing_species.txt", dtype='str', delimiter = '\n')
name_rna = np.loadtxt("../ncbi/species_name_16sRNA_all.fasta", dtype='str', delimiter = '\n')
rna_key_names = name_rna[::2]
print(len(rna_key_names))

for elem in all_species_df:
# for elem in ['Exiguobacterium sp AT1b']:
    # elem = elem.split('_')[:2]
    # elem = elem[0] + ' ' + elem[1]
    # print(elem[0]+' '+elem[1])
    # print ()
    elem = elem.rstrip()

    found = False
    for rna in rna_key_names:
        # print(rna)
        if elem in rna:
            # print("Found",elem, rna)
            found = True
            break
    if not found:
        print("Not found",elem)
