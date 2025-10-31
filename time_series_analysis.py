import numpy as np
import pandas as pd
import time
import json
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

execution_times = {}

fit_data = pd.read_csv(
    './datasets/labelled_regularised_averaged_health_fitness_dataset_testing.csv')
data_dict = {}
pids = np.max(fit_data.participant_id.unique())

# I do this to compute the var_names. This has no effect on the final data_dict
modifiable_fit_data = fit_data.copy()
drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'stress_level', 'date']
for d in drop_cols:
    modifiable_fit_data.drop(d, axis=1, inplace=True)
var_names = modifiable_fit_data.columns

for pid in range(pids):
    # ids start from 1, not 0
    pid += 1
    # I select each participant individually
    fit_data_id = fit_data.loc[fit_data['participant_id'] == pid]
    drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'stress_level', 'date']
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

lpcmci_taus = [2]
lpcmci_pcs = [0.05]
lpcmci_cits = [PairwiseMultCI()]

print('Starting Causal Discovery with PCMCI and LPCMCI\n')

for tau in lpcmci_taus:
    for pc in lpcmci_pcs:
        for cit in lpcmci_cits:
            # LPCMCI
            print('Now executing LPCMCI for tau={} pc={} cit={}...\n'.format(tau, pc, cit))
            start_pc = time.time()
            lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)
            lpcmci_results = lpcmci.run_lpcmci(pc_alpha=pc, tau_max=tau)
            lpcmci_val_matrix = lpcmci_results['val_matrix']
            print('LPCMCI completed for tau={} pc={} cit=PairwiseMultCI\n'.format(tau, pc))

            tp.plot_graph(
                figsize=(18, 12),
                val_matrix=lpcmci_val_matrix,
                graph=lpcmci_results['graph'],
                var_names=var_names,
                arrow_linewidth=5,
                arrowhead_size=150,
                label_fontsize=15,
                tick_label_size=10,
                link_label_fontsize=15,
            )
            plt.title('Causal discovery - LPCMCI with tau={} pc={} cit={}'.format(tau, pc, cit))
            plt.savefig('./graphs/time_series_graphs/TimeSeriesGraph_LPCMCI_tau=' + str(tau) + '_pc=' + str(pc) +
                        '_cit=PairwiseMultCI.pdf')
            plt.close()

            # DoWhy integration
            for i in range(len(var_names)):
                for j in range(len(var_names)):
                    for k in range(len(lpcmci_results['graph'][i][j])):
                        if lpcmci_results['graph'][i][j][k] == 'x-x':
                            lpcmci_results['graph'][i][j][k] = '<->'
                        elif lpcmci_results['graph'][i][j][k] == 'o-o':
                            lpcmci_results['graph'][i][j][k] = '<->'
            lpcmci_graph = create_graph_from_networkx_array(lpcmci_results['graph'], var_names)
            plot(causal_graph=lpcmci_graph, filename='./graphs/time_series_graphs/TimeSeriesGraph_DoWhy_LPCMCI_tau='
                                                     + str(tau) + '_pc=' + str(pc) + '_cit=PairwiseMultCI.pdf',
                 display_plot=False,
                 figure_size=(18, 12))
            execution_times.update({"LPCMCI_tau=" + str(tau) + '_pc=' + str(pc) + '_cit=PairwiseMultCI': str(time.time()- start_pc) + "s"})

pcmci_taus = [2]
pcmci_pcs = [0.1]
pcmci_cits = [PairwiseMultCI()]

for tau in pcmci_taus:
    for pc in pcmci_pcs:
        for cit in pcmci_cits:
            # PCMCI
            print('Now executing PCMCI for tau={} pc={} cit={}...\n'.format(tau, pc, cit))
            start_pc = time.time()
            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)
            pcmci_results = pcmci.run_pcmci(pc_alpha=pc, tau_max=tau)
            pcmci_val_matrix = pcmci_results['val_matrix']
            print('PCMCI completed for tau={} pc={} cit=PairwiseMultCI()\n'.format(tau, pc))

            tp.plot_graph(
                figsize=(18, 12),
                val_matrix=pcmci_val_matrix,
                graph=pcmci_results['graph'],
                var_names=var_names,
                arrow_linewidth=5,
                arrowhead_size=150,
                label_fontsize=15,
                tick_label_size=10,
                link_label_fontsize=15,
            )
            plt.title('Causal discovery - PCMCI with tau={} pc={} cit={}'.format(tau, pc, cit))
            plt.savefig('./graphs/time_series_graphs/TimeSeriesGraph_PCMCI_tau=' + str(tau) + '_pc=' + str(pc) +
                        '_cit=PairwiseMultCI().pdf')
            plt.close()

            # DoWhy integration
            for i in range(len(var_names)):
                for j in range(len(var_names)):
                    for k in range(len(pcmci_results['graph'][i][j])):
                        if pcmci_results['graph'][i][j][k] == 'x-x':
                            pcmci_results['graph'][i][j][k] = '<->'
                        elif pcmci_results['graph'][i][j][k] == 'o-o':
                            pcmci_results['graph'][i][j][k] = '<->'
            pcmci_graph = create_graph_from_networkx_array(pcmci_results['graph'], var_names)
            plot(causal_graph=pcmci_graph, filename='./graphs/'
                                                    'time_series_graphs/TimeSeriesGraph_DoWhy_PCMCI_tau=' + str(tau)
                                                    + '_pc=' + str(pc) + '_cit=PairwiseMultCI().pdf',
                 display_plot=False,
                 figure_size=(18, 12))

            execution_times.update({"PCMCI_tau=" + str(tau) + '_pc=' + str(pc) + '_cit=PairwiseMultCI': str(time.time() - start_pc) + "s"})

print('Causal Discovery with LPCMCI and PCMCI completed\n')

with open('./graphs/causallearn/labelled_execution_times.json', 'a') as f:
    json.dump(execution_times, f)