import csv
import rl_player as rlp

if __name__ == '__main__':
	ratios, scores, comp_ratio = [], [], []

	with open('../'+rlp.SCORES_FILE) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			ratios.append(float(row['ratio']))
			scores.append(float(row['score']))
			if ('completeness_ratio' in row):
				comp_ratio.append(float(row['completeness_ratio']))
	
	import matplotlib.pyplot as plt
	from matplotlib.ticker import MaxNLocator
	plt.figure(1)
	subfig_num = 211
	if (comp_ratio):
		subfig_num += 100
	plt.subplot(subfig_num)
	# plt.plot(ratios)

	plt.title('Evolution of ratio of wins and scores')
	plt.ylabel('Ratio of wins/total games')
	plt.plot(ratios, 'b-')
	plt.grid()
	axes = plt.gca()
	axes.set_ylim([0, 1])
	axes.xaxis.set_major_locator(MaxNLocator(integer=True))
	
	if (comp_ratio):
		subfig_num += 1
		plt.subplot(subfig_num)
		plt.plot(comp_ratio, 'g-')
		plt.grid()
		plt.ylabel('Completeness ratio')
		axes = plt.gca()
		axes.set_ylim([0, 1])
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
