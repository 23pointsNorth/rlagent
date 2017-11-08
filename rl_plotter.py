import csv
import os
import rl_player as rlp

if __name__ == '__main__':
	COMP_ON = os.getenv('SIMP', False)
	ratios, scores, comp_ratio, testing, w_sum, w_sum_std, simp_score = [], [], [], [], [], [], []

	with open('../'+rlp.SCORES_FILE) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			ratios.append(float(row['ratio']))
			scores.append(float(row['score']))
			if ('completeness_ratio' in row):
				comp_ratio.append(float(row['completeness_ratio']))
			if ('testing' in row) and len(row['testing'].strip()) > 0:
				testing.append(float(row['testing']))
			elif testing:
				testing.append(testing[-1])
			if ('weight_sum' in row) and row['weight_sum'] and len(row['weight_sum'].strip()) > 0:
				w_sum.append(float(row['weight_sum']))
				if ('weight_sum_std' in row):
					w_sum_std.append(float(row['weight_sum_std']))
				else:
					w_sum_std.append(.0)
			elif w_sum:
				w_sum.append(w_sum[-1])
				w_sum_std.append(w_sum_std[-1])
			if ('simp_score' in row) and row['simp_score'] and len(row['simp_score'].strip()) > 0:
				simp_score.append(float(row['simp_score']))
			elif simp_score:
				simp_score.append(simp_score[-1])
	
	import matplotlib.pyplot as plt
	from matplotlib.ticker import MaxNLocator
	plt.figure(1, figsize=(7, 8))
	subfig_num = 211
	if (comp_ratio and COMP_ON):
		subfig_num += 100
	if (testing):
		subfig_num += 100
	if (w_sum):
		subfig_num += 100
	if (simp_score):
		subfig_num += 100
	plt.subplot(subfig_num)

	plt.title('Evolution of ratio of wins and scores')
	plt.ylabel('Ratio of wins/total games')
	plt.plot(ratios, 'b-')
	plt.grid()
	axes = plt.gca()
	axes.set_ylim([0, 1])
	axes.xaxis.set_major_locator(MaxNLocator(integer=True))
	
	if (comp_ratio and COMP_ON):
		subfig_num += 1
		plt.subplot(subfig_num)
		plt.plot(comp_ratio, 'g-')
		plt.grid()
		plt.ylabel('Completeness ratio')
		axes = plt.gca()
		axes.set_ylim([0, 1])
		axes.xaxis.set_major_locator(MaxNLocator(integer=True))

	if (testing):
		subfig_num += 1
		plt.subplot(subfig_num)
		plt.plot(testing, 'c-')
		plt.grid()
		plt.ylabel('Testing (argmax)')
		axes = plt.gca()
		axes.set_ylim([0, 1])
		axes.xaxis.set_major_locator(MaxNLocator(integer=True))

	if (w_sum):
		subfig_num += 1
		plt.subplot(subfig_num)
		x = range(0, len(w_sum))
		plt.errorbar(x, w_sum, yerr=w_sum_std, 
			color='m', linestyle='-', capsize=2, 
			errorevery=int(max(1, len(w_sum)/50)))
		plt.grid()
		plt.ylabel('Avg Weight')
		axes = plt.gca()
		axes.set_ylim([0, 1])
		axes.xaxis.set_major_locator(MaxNLocator(integer=True))

	if (simp_score):
		subfig_num += 1
		plt.subplot(subfig_num)
		plt.plot(simp_score, 'c-')
		plt.grid()
		plt.ylabel('Simp Scores')
		axes = plt.gca()
		axes.xaxis.set_major_locator(MaxNLocator(integer=True))

	# Plot scores
	subfig_num += 1
	plt.subplot(subfig_num)
	plt.plot(scores, 'r-')
	plt.grid()
	plt.ylabel('Average Scores')
	plt.xlabel('Training Epochs (40 trajectories each)')
	axes = plt.gca()
	axes.xaxis.set_major_locator(MaxNLocator(integer=True))


	plt.show()
