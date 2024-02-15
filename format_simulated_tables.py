import pandas as pd
import pdb

final_table = pd.DataFrame()

for distortion in ["0.005", "0.01", "0.02", "0.1"]:
	for n, k in [
		(100, 10), (1000, 1), (25, 100), (100, 25),
		(500, 5), (50, 100), (1000, 5), (100, 100),
		(1000, 10), (250, 100), (1000, 25), (500, 100),
		(1000, 50)
	]:
		experiment_results = pd.read_csv(f"data/results_N={n}_K={k}_responses_simulated_distr_dist={distortion}_gen_N=1000_K=100_n_samples=1000.json.csv")
		experiment_results = experiment_results[experiment_results.Sampler == '(bootstrap_items,sample_all)'].reset_index()
		intermediate_table = pd.DataFrame()
		intermediate_table['Metric'] = ['$\\Gamma_{\\rm MAE}$', '$\\Gamma_{\\rm WINS}$', '$\\Gamma_{\\rm MEMD}$']
		intermediate_table['N'] = n
		intermediate_table['K'] = k
		intermediate_table['$\\epsilon$'] = distortion
		intermediate_table['$\\Delta$'] = (experiment_results['M2 GT Alt'] - experiment_results['M1 GT Alt']).abs()
		intermediate_table['p'] = experiment_results['GT P-score']
		intermediate_table['num_ratings'] = n*k
		final_table = pd.concat([final_table, intermediate_table])

for metric in ['$\\Gamma_{\\rm MAE}$', '$\\Gamma_{\\rm WINS}$', '$\\Gamma_{\\rm MEMD}$']:
	metric_table = final_table[final_table.Metric == metric]
	#metric_table = metric_table.melt(id_vars = ['num_ratings', 'N', 'K', '$\\epsilon$'], value_vars=['$\\Delta$', 'p'])
	#metric_table = metric_table.sort_values(['num_ratings', 'N', 'K', '$\\epsilon$', 'variable'])
	metric_table = metric_table.sort_values(['num_ratings', 'N', 'K', '$\\epsilon$'])
	#pdb.set_trace()
	#metric_table = metric_table.drop(columns='num_ratings')
	#metric_table = metric_table.pivot(index=['num_ratings','N','K', 'variable'], columns=['$\\epsilon$', 'variable'], values=['value'])
	#metric_table.index = metric_table.index.droplevel()
	metric_table = metric_table.pivot(index=['num_ratings','N','K'], columns=['$\\epsilon$'], values=['$\\Delta$', 'p'])
	metric_table.index = metric_table.index.droplevel()
	metric_table = metric_table.style.format(precision=4)

	print(metric)
	print()
	print(metric_table.to_latex())
