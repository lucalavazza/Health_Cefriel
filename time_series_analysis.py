import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.lpcmci import LPCMCI
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.pairwise_CI import PairwiseMultCI
from tigramite.independence_tests.parcorr import ParCorr
from dowhy.utils.timeseries import create_graph_from_networkx_array
from dowhy.utils.plotting import plot

# I want to avoid some warnings
pd.options.mode.chained_assignment = None

fit_data = pd.read_csv(
    './datasets/labelled_regularised_averaged_health_fitness_dataset_testing.csv')
data_dict = {}
pids = np.max(fit_data.participant_id.unique())

# I do this to compute the var_names. This has no effect on the final data_dict
modifiable_fit_data = fit_data.copy()
drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'stress_level']
for d in drop_cols:
    modifiable_fit_data.drop(d, axis=1, inplace=True)
var_names = modifiable_fit_data.columns

for pid in range(pids):
    # ids start from 1, not 0
    pid += 1
    # I select each participant individually
    fit_data_id = fit_data.loc[fit_data['participant_id'] == pid]
    drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'stress_level']
    for d in drop_cols:
        fit_data_id.drop(d, axis=1, inplace=True)
    fit_data_id.reset_index(drop=True, inplace=True)
    data_array_pid = fit_data_id.to_numpy()
    data_dict_pid = {}
    month = []
    for i in range(len(data_array_pid)):
        columns = []
        for j in range(len(var_names)):
            columns.append(data_array_pid[i][j])
        month.append(columns)
    data_dict_pid.update({pid: np.array(month)})
    data_dict.update(data_dict_pid)

dataframe = pp.DataFrame(data_dict, analysis_mode='multiple', var_names=var_names)

taus = [2]
pcs = [0.05]
cits = [PairwiseMultCI()]

print('Starting Causal Discovery with PCMCI and LPCMCI\n')

for tau in taus:
    for pc in pcs:
        for cit in cits:
            # LPCMCI
            print('Now executing LPCMCI for tau={} pc={} cit={}...\n'.format(tau, pc, cit))

            lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)
            results = lpcmci.run_lpcmci(pc_alpha=pc, tau_max=tau)
            val_matrix = results['val_matrix']

            print('LPCMCI completed for tau={} pc={} cit={}\n'.format(tau, pc, cit))

            tp.plot_graph(
                figsize=(18, 12),
                val_matrix=val_matrix,
                graph=results['graph'],
                var_names=var_names,
                arrow_linewidth=5,
                arrowhead_size=150,
                label_fontsize=15,
                tick_label_size=10,
                link_label_fontsize=15,
            )

            plt.title('Causal discovery - LPCMCI with tau={} pc={} cit={}'.format(tau, pc, cit))

            plt.savefig(
                './graphs/time_series_graphs/TimeSeriesGraph_LPCMCI_tau='
                + str(tau) + '_pc=' + str(pc) + '_cit=' + str(cit) + '.pdf')
            plt.close()

            # DoWhy Integration
            print('Now executing PCMCI for tau={} pc={} cit={}...\n'.format(tau, pc, cit))
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
            results_pcmci = pcmci.run_pcmci(pc_alpha=pc, tau_max=tau)
            print('PCMCI completed for tau={} pc={} cit={}\n'.format(tau, pc, cit))
            for i in range(len(var_names)):
                for j in range(len(var_names)):
                    for k in range(len(results_pcmci['graph'][i][j])):
                        if results_pcmci['graph'][i][j][k] == 'x-x':
                            results_pcmci['graph'][i][j][k] = '<->'
                        elif results_pcmci['graph'][i][j][k] == 'o-o':
                            results_pcmci['graph'][i][j][k] = '<->'
            graph = create_graph_from_networkx_array(results_pcmci['graph'], var_names)
            plot(causal_graph=graph, filename='./graphs/'
                                              'time_series_graphs/TimeSeriesGraph_DoWhy_PCMCI_tau=' + str(tau) + '_pc='
                                              + str(pc) + '_cit=' + str(cit) + '.pdf', display_plot=False,
                 figure_size=(18, 12))

print('Causal Discovery with LPCMCI completed\n')
