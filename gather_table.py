import pandas as pd

final_table = pd.DataFrame()
for distortion in ['0.005', '0.01', '0.02', '0.1']:
	experiment_results = pd.read_csv(f'results_N=168_K=6_responses_simulated_distr_dist={distortion}_gen_N=168_K=6_num_samples=1000.json.csv')
	intermediate_table = pd.DataFrame()
	intermediate_table['$\\Delta$'] = experiment_results['M2 GT Alt'] - experiment_results['M1 GT Alt']
	intermediate_table['p-value'] = experiment_results['GT_Pvalue']
	intermediate_table['Metric'] = ['$\\Gamma_{MAE}$', '$\\Gamma_{MEMD}$', '$\\Gamma_{WINS}$']
	intermediate_table[f'$\\epsilon$'] = distortion
	final_table = pd.concat([final_table, intermediate_table])
final_table = final_table.pivot(index='$\\epsilon$', columns='Metric')
final_table = final_table.reset_index(drop=True)
print(final_table.to_latex(index=False))
