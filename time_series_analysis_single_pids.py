import numpy as np
import os
import pandas as pd
import tigramite
import warnings
from matplotlib import pyplot as plt
from numpy.ma.extras import average
from numpy.random import SeedSequence
from numpy.ma.core import shape
from sklearn.preprocessing import LabelEncoder
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.pairwise_CI import PairwiseMultCI
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.parcorr import ParCorr

# I want to avoid some warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings(action='ignore', category=UserWarning)

# I need this just to get the pids
fit_data = pd.read_csv(
    '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/labelled_regularised_averaged_health_fitness_dataset.csv')

pids = np.max(fit_data.participant_id.unique())

# I do this on the whole dataset just to compute the var_names. This has no effect on the final data_dict.
drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'age', 'date']
for d in drop_cols:
    fit_data.drop(d, axis=1, inplace=True)
var_names = fit_data.columns

c_pids = [1]

for pid in c_pids:
    data_dict = {}
    data_dict_pid = {}

    fit_data_pid = pd.read_csv('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/participants/pid-'+str(pid+1)+'/health_fitness_dataset_pid-'+str(pid+1)+'.csv')

    # numerical encoding
    fit_data_pid.replace(['F', 'M', 'Others'], [0, 1, 2], inplace=True)
    fit_data_pid.replace(['Never', 'Current', 'Former'], [0, 1, 2], inplace=True)
    fit_data_pid.replace(['None', 'Hypertension', 'Diabetes', 'Asthma'], [0, 1, 2, 3], inplace=True)
    non_numeric_columns = list(fit_data_pid.select_dtypes(exclude=[np.number]).columns)
    label = LabelEncoder()
    for col in non_numeric_columns:
        if col not in ['date', 'gender']:
            fit_data_pid[col] = label.fit_transform(fit_data_pid[col])

    days = len(fit_data_pid.date.unique())
    day = []

    # I can drop the date as well, because the temporal order it is later computed via the index
    drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'age', 'date']
    for d in drop_cols:
        fit_data_pid.drop(d, axis=1, inplace=True)
    fit_data_pid.reset_index(drop=True, inplace=True)

    data_array_pid = fit_data_pid.to_numpy()

    for i in range(len(data_array_pid)):
        columns = []
        for j in range(len(var_names)):
            columns.append(data_array_pid[i][j])
        day.append(columns)
    data_dict_pid.update({pid: np.array(day)})
    data_dict.update(data_dict_pid)

    dataframe = pp.DataFrame(data_dict, analysis_mode='multiple', var_names=var_names)

    taus = [1, 2, 3, 4, 5]
    pcs = [0.02, 0.05, 0.07]
    cits = [ParCorr()]

    for tau in taus:
        for pc in pcs:
            for cit in cits:
                # LPCMCI
                print('Now executing LPCMCI for pid={}, tau={},  pc={},  cit={}...\n'.format(pid, tau, pc, cit))

                lpcmci = LPCMCI(dataframe=dataframe, cond_ind_test=cit, verbosity=0)
                results = lpcmci.run_lpcmci(pc_alpha=pc, tau_max=tau)
                val_matrix = results['val_matrix']

                print('LPCMCI completed for pid={}, tau={}, pc={}, cit={}\n'.format(pid, tau, pc, cit))

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

                plt.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/time_series_graphs/tsg_pids/pid='
                            + str(pid) + '_TimeSeriesGraph_LPCMCI_tau=' + str(tau) + '_pc=' + str(pc) + '_cit=' + str(cit) + '.png')
                plt.close()
