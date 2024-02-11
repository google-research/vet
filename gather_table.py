import pandas as pd
from absl import app
from absl import flags
from typing import Sequence

_K_RESPONSES = flags.DEFINE_integer(
    "k_responses", 5, "Number of responses per item."
)
_N_ITEMS = flags.DEFINE_integer(
    "n_items", 500, "Number of sample sets per experiment."
)
_DISTORTION = flags.DEFINE_list(
    "distortion", ['0.3', '0.35', '0.37', '0.39', '0.4'], "Number of sample sets per experiment."
)

def main(argv: Sequence[str]) -> None:
	final_table = pd.DataFrame()
	for distortion in _DISTORTION.value:
		experiment_results = pd.read_csv(f'results_N={_N_ITEMS.value}_K={_K_RESPONSES.value}_responses_simulated_distr_dist={distortion}_gen_N={_N_ITEMS.value}_K={_K_RESPONSES.value}_num_samples=1000.json.csv')
		intermediate_table = pd.DataFrame()
		intermediate_table['$\\Delta$'] = (experiment_results['M2 GT Alt'] - experiment_results['M1 GT Alt']).abs()
		intermediate_table['p-value'] = experiment_results['GT_Pvalue']
		intermediate_table['Metric'] = ['$\\Gamma_{MAE}$', '$\\Gamma_{MEMD}$', '$\\Gamma_{WINS}$']
		intermediate_table[f'$\\epsilon$'] = distortion
		final_table = pd.concat([final_table, intermediate_table])
	#final_table = final_table.pivot(index='$\\epsilon$', columns='Metric')
	final_table = final_table.melt(["Metric", "$\\epsilon$"]).sort_values(by=["Metric","variable"]).pivot(index = "$\\epsilon$", columns=["Metric","variable"])
	final_table = final_table.reset_index(drop=True)
	print(final_table.to_latex(index=False))

if __name__ == "__main__":
  app.run(main)

