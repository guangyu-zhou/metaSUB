def extract_vec():
	extracted_set = {}
	spec_vocab = set()
	sample_vec = {}
	for i in range(1748535, 1750107,1):
		if i == 1749159 or i == 1750038:
				continue
		s1 = extract_species("../profile_merged/profiled_SRR" + str(i) +".txt")
		# print(set(s1.keys()))
		spec_vocab |= set(s1.keys())
		# print(len(spec_vocab), len(s1))
		extracted_set['SRR'+str(i)] = s1
	return extracted_set, spec_vocab

def extract_vec_full():
	extracted_set = {}
	all_species = set()
	sample_vec = {}
	for i in range(1748535, 1750107,1):
		if i == 1749159 or i == 1750038:
				continue
		s1 = extract_species("../profile_merged/profiled_SRR" + str(i) +".txt")
		all_species |= set(s1.keys())
		extracted_set['SRR'+str(i)] = s1

	all_species = list(all_species)
	# print(all_species[0])
	print("number of samples",len(extracted_set), "number of species",len(all_species))
	vec_full = {}
	vec_list = []
	vec_key = []
	cnt = 0
	for _samp, _spec in extracted_set.iteritems():
		# print(_samp, len(_spec))

		result = []

		for elem in all_species:
			if elem in _spec.keys():
				result.append(_spec[elem]/100)
			else:
				result.append(0.0)
		result = np.array(result)	
		
		vec_full[_samp] = result
		vec_list.append(result)
		vec_key.append(_samp)
		# print(len(np.where(result > 0.0)[0]))

		cnt+=1
		if cnt > 500:
			break
	# vec_full = np.array(vec_full)
	# print(vec_full, type(vec_full))
	print(vec_key)
	return vec_full, vec_list, vec_key
	
def plot_by_sample():
	res = load_meta()
	vec, vec_list, vec_key = extract_vec_full()
	# print(len(vec_list), len(vec_list[0]))

	samples = vec_key
	sim_mat = compute_pairwise_sim(vec_list)

	x = []
	y = []
	for i in range(len(samples) - 1):
		if res[samples[i]] == 'not applicable':
			continue
		for j in range(i+1, len(samples)):
			print(i,j, samples[i], samples[j])
			# print(res[samples[j]])
			if res[samples[j]] == 'not applicable':
				continue
			# print(res[samples[i]], res[samples[j]])
			dist = geo_dist(res[samples[i]].split(' ')[0],res[samples[i]].split(' ')[2], res[samples[j]].split(' ')[0],res[samples[j]].split(' ')[2])

			if dist > 500:
				# print("Outlier, skip", res[samples[i]], res[samples[j]])
				continue
			sim = sim_mat[i][j]
			x.append(dist)
			y.append(sim)
			# print(res[sample].split(' ')[0],res[sample].split(' ')[2], vec[sample][:100])


	import matplotlib.pyplot as plt
	import pylab
	# from scipy.optimize import curve_fit
	# x.sort()
	# plt.plot(x, 'o')
	# plt.show()

	print(len(x), len(y))
	# print(x)
	# print(y)
	plt.scatter(x, y, label = 'data', s = 0.2)
	plt.xlabel('dist')
	plt.ylabel('cos sim')
	plt.legend()
	plt.show()
