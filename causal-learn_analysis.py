import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dowhy import CausalModel
import numpy as np
import csv
import os
from sklearn.preprocessing import LabelEncoder
from causallearn.utils.cit import *
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss
from causallearn.search.Granger.Granger import Granger
from causallearn.utils.GraphUtils import GraphUtils

# create the directories for the causal graphs
graphs_dir = '/Users/luca_lavazza/Documents/GitHub/Health_Cefriel/graphs/'
try:
    os.mkdir(graphs_dir)
except FileExistsError:
    pass
except PermissionError:
    print(f"Permission denied: Unable to create '" + dir_name + "'.")
except Exception as e:
    print(f"An error occurred: {e}")

fit_data = pd.read_csv('averaged_health_fitness_dataset.csv')
drop_cols = ['participant_id', 'gender', 'date', 'height_cm', 'weight_kg', 'activity_type']
fit_data = fit_data.drop(drop_cols, axis=1)
var_names = fit_data.columns
fit_data = fit_data.to_numpy()

# CONSTRAINT-BASED METHODS
# PC
cg_pc = pc(fit_data)
pyd_pc = GraphUtils.to_pydot(cg_pc.G, labels=var_names)
pyd_pc.write_png(graphs_dir+'causal_graph_causal-learn_pc.png')
# FCI
cg_fci, edges_fci = fci(fit_data)
pyd_fci = GraphUtils.to_pydot(cg_fci, labels=var_names)
pyd_fci.write_png(graphs_dir+'causal_graph_causal-learn_fci.png')

# Generalized Independence Noise (GIN) condition-based method -> it's like a Black Sabbath's riff: slow and heavy
# G, K = GIN(fit_data)
# pyd_GIN = GraphUtils.to_pydot(G, labels=var_names)
# tmp_png = pyd_GIN.create_png(f="png")
# fp = io.BytesIO(tmp_png)
# img = mpimg.imread(fp, format='png')
# plt.axis('off')
# plt.imshow(img)
# plt.show()

# Permutation-based causal discovery methods
# GRaSP
cd_grasp = grasp(fit_data)
pyd_grasp = GraphUtils.to_pydot(cd_grasp, labels=var_names)
pyd_fci.write_png(graphs_dir+'causal_graph_causal-learn_grasp.png')
# BOSS
cd_boss = boss(fit_data)
pyd_boss = GraphUtils.to_pydot(cd_boss, labels=var_names)
pyd_boss.write_png(graphs_dir+'causal_graph_causal-learn_boss.png')

