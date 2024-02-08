import pandas as pd

intermediate_table = pd.DataFrame()
for distortion in ['0.005', '0.01', '0.02', '0.1']:
	experiment_results = pd.read_csv(f'results_N=168_K=5_responses_simulated_distr_dist={distortion}_gen_N=168_K=5_num_samples=1000.json.csv')
	intermediate_table[(f'$\\epsilon = {distortion}$', '$A$')] = experiment_results['M1 GT Alt']
	intermediate_table[(f'$\\epsilon = {distortion}$', '$B$')] = experiment_results['M2 GT Alt']
	intermediate_table[(f'$\\epsilon = {distortion}$', '$\\Delta$')] = experiment_results['M2 GT Alt'] - experiment_results['M1 GT Alt']

intermediate_table.index = ['$\\Gamma_{MAE}$', '$\\Gamma_{WINS}$', '$\\Gamma_{MEMD}$']
final_table = intermediate_table.transpose()
final_table.index = pd.MultiIndex.from_tuples(final_table.index)
means = final_table.query("ilevel_1 == '$A$'").apply(['mean'])
means.index = [('','$A$')]
final_table = final_table.drop('$A$', level=1)
final_table = pd.concat([means, final_table])
final_table.index = pd.MultiIndex.from_tuples(final_table.index)
print(final_table.to_latex())
