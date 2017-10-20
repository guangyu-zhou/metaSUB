from __future__ import print_function

import numpy as np
import pandas as pd
# import gmplot
import pickle
from gmplot import GoogleMapPlotter as gmp
from gmplot import color_dicts
def plot_location_node(locs, labels):
	mymap = gmp.from_geocode("New York")

	clist = color_dicts.html_color_codes.keys()
	for cur_label in range(np.max(labels)):
		print("cur_label",cur_label)
		path = [[],[]]
		color = ''
		for i in range(len(locs)):
			loc = locs[i]
			cluster_label = labels[i]
			assert(loc !=  'not applicable')
			if cluster_label == cur_label:
				# print("Color", clist[cluster_label])
				color = clist[cluster_label]
				path[0].append(float(loc.split(' ')[0]))
				path[1].append(-float(loc.split(' ')[2]))
		
		edge = [tuple(path[0]),tuple(path[1])]
		# mymap.heatmap(edge[0], edge[1], threshold=5, radius=40)
		mymap.scatter(edge[0], edge[1], c = color,s=200, marker=False, alpha=1)
	mymap.draw("cluster_map.html")

def index():
	fin_loc = open("../files/loc_index.txt", 'r')
	locs = []
	for line in fin_loc:
		locs.append(line)

	labels = []
	fin_label = open('../plots/heatmap_out/cluster_label.txt')
	first_line = True
	for line in fin_label:
		if first_line:
			first_line = False
			continue
		label = int(line.split()[1])
		labels.append(label)
	assert(len(locs) == len(labels))
	return locs, labels

	# print(loc_index)
locs, labels = index()
plot_location_node(locs, labels)

