import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.parcorr import ParCorr

# I want to avoid some warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action='ignore', category=UserWarning)

# I need this just to get the pids
fit_data = pd.read_csv(
    '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/labelled_regularised_averaged_health_fitness_dataset.csv')

pids = np.max(fit_data.participant_id.unique())

c_pids = range(1, pids+1)

for pid in c_pids:
    data_dict = {}
    data_dict_pid = {}

    fit_data_pid = fit_data.loc[fit_data['participant_id'] == pid]

    days = len(fit_data_pid.date.unique())
    day = []

    # I can drop the date as well, because the temporal order it is later computed via the index
    drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'bmi', 'resting_heart_rate',
                 'blood_pressure_systolic', 'blood_pressure_diastolic', 'smoking_status', 'health_condition']
    for d in drop_cols:
        fit_data_pid.drop(d, axis=1, inplace=True)
    fit_data_pid.reset_index(drop=True, inplace=True)

    var_names = fit_data_pid.columns

    data_array_pid = fit_data_pid.to_numpy()

    for i in range(len(data_array_pid)):
        columns = []
        for j in range(len(var_names)):
            columns.append(data_array_pid[i][j])
        day.append(columns)
    data_dict_pid.update({pid: np.array(day)})
    data_dict.update(data_dict_pid)

    dataframe = pp.DataFrame(data_dict, analysis_mode='multiple', var_names=var_names)

    taus = [3]
    pcs = [0.05]
    cits = [ParCorr()]

    for tau in taus:
        for pc in pcs:
            for cit in cits:
                # LPCMCI
                print('Now executing LPCMCI for pid={}, tau={},  pc={},  cit={}...\n'.format(pid, tau, pc, cit))

                lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)
                results = lpcmci.run_lpcmci(pc_alpha=pc, tau_max=tau)
                val_matrix = results['val_matrix']

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

                plt.title('Causal discovery - LPCMCI for pid={} with tau={}, pc={}, cit={}'.format(pid, tau, pc, cit))

                plt.savefig(
                    '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/time_series_graphs/tsg_pids/pid='
                    + str(pid) + '_TimeSeriesGraph_LPCMCI_tau=' + str(tau) + '_pc=' + str(pc) + '_cit=' + str(
                        cit) + '.png')
                plt.close()
