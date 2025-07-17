import numpy as np
import os

import pandas as pd
import tigramite
from matplotlib import pyplot as plt
from numpy.ma.extras import average
from numpy.random import SeedSequence
from numpy.ma.core import shape
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.pairwise_CI import PairwiseMultCI

pd.options.mode.chained_assignment = None

fit_data = pd.read_csv(
    '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/datasets/labelled_regularised_averaged_health_fitness_dataset.csv')

data_dict = {}

pids = np.max(fit_data.participant_id.unique())

for pid in range(pids):
    pid += 1
    fit_data_id = fit_data.loc[fit_data['participant_id'] == pid]
    drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'age', 'date']
    for d in drop_cols:
        fit_data_id.drop(d, axis=1, inplace=True)
    fit_data_id.reset_index(drop=True, inplace=True)
    data_array_pid = fit_data_id.to_numpy()
    data_dict_pid = {}
    month = []
    for i in range(len(data_array_pid)):
        columns = []
        for j in range(12):
            columns.append(data_array_pid[i][j])
        month.append(columns)
    data_dict_pid.update({pid: np.array(month)})
    data_dict.update(data_dict_pid)

drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'age', 'date']
for d in drop_cols:
    fit_data.drop(d, axis=1, inplace=True)
var_names = fit_data.columns

dataframe = pp.DataFrame(data_dict, analysis_mode='multiple', var_names=var_names)

pcmci = PCMCI(dataframe=dataframe, cond_ind_test=PairwiseMultCI(), verbosity=0)
results = pcmci.run_pcmci()
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

plt.savefig('/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/TimeSeriesGraph.png')
plt.close()
