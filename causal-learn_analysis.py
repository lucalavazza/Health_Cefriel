from causallearn.graph.GraphClass import *
from causallearn.utils.cit import *
from causallearn.score.LocalScoreFunction import *
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.Granger.Granger import Granger
import time

np.random.seed(42)

# create the directories for the causal graphs
graphs_dir = './graphs/causallearn/graphs'
npy_dir = './graphs/causallearn/edges/npy'
txt_dir = './graphs/causallearn/edges/txt'
try:
    os.mkdir(graphs_dir)
    os.mkdir(npy_dir)
    os.mkdir(txt_dir)
except FileExistsError:
    pass
except PermissionError:
    print(f"Permission denied: Unable to create the directory.")
except Exception as e:
    print(f"An error occurred: {e}")

execution_times = {}

data_type = 'labelled'
# data_type = 'encoded'
fit_data = pd.read_csv('./datasets/' + data_type + '_regularised_averaged_health_fitness_dataset_training.csv')
if data_type == 'encoded':
    drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender_M', 'gender_F', 'gender_Other', 'stress_level']
else:
    drop_cols = ['participant_id', 'height_cm', 'weight_kg', 'gender', 'stress_level']
fit_data = fit_data.drop(drop_cols, axis=1)
var_names = fit_data.columns
fit_data = fit_data.to_numpy()

# Constraint-based Methods
# PC
# cits = ['fisherz', 'chisq', 'gsq', 'kci']
# TODO: try chisq
cits = ['fisherz', 'gsq', 'chisq']  # chisq takes A LOT (~40 mins), while kci was still computing after 16h: decided to skip
for cit in cits:
    print('---> PC, alpha=0.05, cit = ' + str(cit) + ', uc_rule: 0, uc_priority: 0, no bg_knowledge')
    start_pc = time.time()
    cg_pc = pc(fit_data, alpha=0.05, indep_test=cit, uc_rule=0, uc_priority=0)
    print("PC with cit = " + str(cit) + ": " + str((time.time() - start_pc)) + " seconds\n\n")
    execution_times.update({data_type + ", PC - " + cit: str(time.time() - start_pc) + "s"})
    pcg_pc = GraphUtils.to_pydot(cg_pc.G, labels=var_names)
    if data_type == 'encoded':
        pcg_pc.write_png(
            graphs_dir + '/onehot/PC-onehot/encoding_causal_graph_causal-learn_pc_' + str(cit) + '.png')
    else:
        pcg_pc.write_png(graphs_dir + '/labelling/PC-labelling/labelling_causal_graph_causal-learn_pc_' + str(cit)
                         + '.png')

    np_pc_edges = np.array(cg_pc.find_fully_directed())
    np_pc_nodes = []
    for edge in np_pc_edges:
        for node in edge:
            if node not in np_pc_nodes:
                np_pc_nodes.append(node)
    np_pc_names = var_names.to_numpy()
    np_pc_edges_names = []
    np_pc_nodes_names = []
    for edge in np_pc_edges:
        new_edge = []
        for node in edge:
            new_node = var_names[node]
            new_edge.append(new_node)
        np_pc_edges_names.append(new_edge)
    for node in np_pc_nodes:
        np_pc_nodes_names.append(var_names[node])
    np_pc_edges_names = np.array(np_pc_edges_names)

    if data_type == 'encoded':
        np.save(npy_dir + '/encoding_causal_graph_causal-learn_pc_' + str(cit) + '.npy', np_pc_edges_names)
        with open(txt_dir + '/encoding_causal_graph_causal-learn_pc_' + str(cit) + '.txt', 'w') as f:
            for edge in np_pc_edges_names:
                f.write(f"{edge}\n")
    else:
        np.save(npy_dir + '/labelling_causal_graph_causal-learn_pc_' + str(cit) + '.npy', np_pc_edges_names)
        with open(txt_dir + '/labelling_causal_graph_causal-learn_pc_' + str(cit) + '.txt', 'w') as f:
            for edge in np_pc_edges_names:
                f.write(f"{edge}\n")
    # FCI
    print('---> FCI, alpha=0.05, cit = ' + str(cit) + ', no bg_knowledge')
    start_fci = time.time()
    cg_fci, fci_edges = fci(fit_data, alpha=0.05, indep_test=cit, node_names=var_names)
    print("FCI with cit = " + str(cit) + ": " + str((time.time() - start_fci)) + " seconds\n\n")
    execution_times.update({data_type + ", FCI - " + cit: str(time.time() - start_fci) + "s"})
    pcg_fci = GraphUtils.to_pydot(cg_fci, labels=var_names)
    if data_type == 'encoded':
        pcg_fci.write_png(graphs_dir + '/onehot/FCI-onehot/encoding_causal_graph_causal-learn_fci_' + str(cit) + '.png')
    else:
        pcg_fci.write_png(graphs_dir + '/labelling/FCI-labelling/labelling_causal_graph_causal-learn_fci_' + str(cit)
                          + '.png')

    np_fci_edges = []
    for i in range(len(fci_edges)):
        new_edge = [str(fci_edges[i].get_node1()), str(fci_edges[i].get_node2())]
        np_fci_edges.append(new_edge)

    if data_type == 'encoded':
        np.save(npy_dir + '/encoding_causal_graph_causal-learn_fci_' + str(cit) + '.npy', np_fci_edges)
        with open(txt_dir + '/encoding_causal_graph_causal-learn_fci_' + str(cit) + '.txt', 'w') as f:
            for edge in np_fci_edges:
                f.write(f"{edge}\n")
    else:
        np.save(npy_dir + '/labelling_causal_graph_causal-learn_fci_' + str(cit) + '.npy', np_fci_edges)
        with open(txt_dir + '/labelling_causal_graph_causal-learn_fci_' + str(cit) + '.txt', 'w') as f:
            for edge in np_fci_edges:
                f.write(f"{edge}\n")

# Score-based Method
# GES
if data_type != 'encoded':  # does not seem to work with encoded data --> skipped
    score_funcs = ['local_score_BIC']  # local_score_BDeu does not start --> skipped
    for sf in score_funcs:
        print('---> GES, score function = ' + str(sf))
        start_pc = time.time()
        cg_ges = ges(fit_data, score_func=sf)
        pcg_ges = GraphUtils.to_pydot(cg_ges['G'], labels=var_names)
        print("GES with score_func = " + str(sf) + ": " + str((time.time() - start_pc)) + " seconds\n\n")
        execution_times.update({data_type + ", GES - " + sf: str(time.time() - start_pc) + "s"})
        if data_type == 'encoded':
            pcg_ges.write_png(
                graphs_dir + '/onehot/GES-onehot/encoding_causal_graph_causal-learn_ges_' + str(sf) + '_.png')
        else:
            pcg_ges.write_png(
                graphs_dir + '/labelling/GES-labelling/labelling_causal_graph_causal-learn_ges_' + str(sf) + '_.png')

        ges_edges = cg_ges['G'].get_graph_edges()
        np_ges_edges = []
        for i in range(len(ges_edges)):
            edge = [int(str(ges_edges[i].get_node1()).replace('X', ''))-1, int(str(ges_edges[i].get_node2()).replace('X', ''))-1]
            np_ges_edges.append(np.array(edge))

        np_ges_nodes = []
        for edge in np_ges_edges:
            for node in edge:
                if node not in np_ges_nodes:
                    np_ges_nodes.append(node)
        np_ges_names = var_names.to_numpy()
        np_ges_edges_names = []
        np_ges_nodes_names = []
        for edge in np_ges_edges:
            new_edge = []
            for node in edge:
                new_node = var_names[node]
                new_edge.append(new_node)
            np_ges_edges_names.append(new_edge)
        for node in np_ges_nodes:
            np_ges_nodes_names.append(var_names[node])
        np_ges_edges_names = np.array(np_ges_edges_names)

        if data_type == 'encoded':
            np.save(npy_dir + '/encoding_causal_graph_causal-learn_ges_' + str(sf) + '.npy', np_ges_edges_names)
            with open(txt_dir + '/encoding_causal_graph_causal-learn_ges_' + str(sf) + '.txt', 'w') as f:
                for edge in np_ges_edges_names:
                    f.write(f"{edge}\n")
        else:
            np.save(npy_dir + '/labelling_causal_graph_causal-learn_ges_' + str(sf) + '.npy', np_ges_edges_names)
            with open(txt_dir + '/labelling_causal_graph_causal-learn_ges_' + str(sf) + '.txt', 'w') as f:
                for edge in np_ges_edges_names:
                    f.write(f"{edge}\n")

# Functional Causal Models
# LiNGAM
print('---> LiNGAM')  # quite slow with encoded data
start_pc = time.time()
model = lingam.ICALiNGAM(random_state=7, max_iter=1000)
model.fit(fit_data)
adj_matr = model.adjacency_matrix_
print("LiNGAM: " + str((time.time() - start_pc)) + " seconds\n\n")
execution_times.update({data_type + ", LiNGAM": str(time.time() - start_pc) + "s"})

num_nodes = adj_matr.shape[0]
cg_lin = CausalGraph(num_nodes)
for i in range(num_nodes):
    for j in range(num_nodes):
        edge1 = cg_lin.G.get_edge(cg_lin.G.nodes[i], cg_lin.G.nodes[j])
        if edge1 is not None:
            cg_lin.G.remove_edge(edge1)

for i in range(num_nodes):
    for j in range(num_nodes):
        if adj_matr[i, j] > 0:
            cg_lin.G.add_edge(Edge(cg_lin.G.nodes[i], cg_lin.G.nodes[j], Endpoint.TAIL, Endpoint.ARROW))
        elif adj_matr[i, j] < 0:
            cg_lin.G.add_edge(Edge(cg_lin.G.nodes[j], cg_lin.G.nodes[i], Endpoint.TAIL, Endpoint.ARROW))

pcg_lin = GraphUtils.to_pydot(cg_lin.G, labels=var_names)
if data_type == 'encoded':
    pcg_lin.write_png(graphs_dir + '/onehot/LiNGAM-onehot/encoding_causal_graph_causal-learn_lingam.png')
else:
    pcg_lin.write_png(graphs_dir + '/labelling/LiNGAM-labelling/labelling_causal_graph_causal-learn_lingam.png')

np_lin_edges = np.array(cg_lin.find_fully_directed())
np_lin_nodes = []
for edge in np_lin_edges:
    for node in edge:
        if node not in np_lin_nodes:
            np_lin_nodes.append(node)
np_lin_names = var_names.to_numpy()
np_lin_edges_names = []
np_lin_nodes_names = []
for edge in np_lin_edges:
    new_edge = []
    for node in edge:
        new_node = var_names[node]
        new_edge.append(new_node)
    np_lin_edges_names.append(new_edge)
for node in np_lin_nodes:
    np_lin_nodes_names.append(var_names[node])
np_lin_edges_names = np.array(np_lin_edges_names)

if data_type == 'encoded':
    np.save(npy_dir + '/encoding_causal_graph_causal-learn_lingam.npy', np_lin_edges_names)
    with open(txt_dir + '/encoding_causal_graph_causal-learn_lingam.txt', 'w') as f:
        for edge in np_lin_edges_names:
            f.write(f"{edge}\n")
else:
    np.save(npy_dir + '/labelling_causal_graph_causal-learn_lingam.npy', np_lin_edges_names)
    with open(txt_dir + '/labelling_causal_graph_causal-learn_lingam.txt', 'w') as f:
        for edge in np_lin_edges_names:
            f.write(f"{edge}\n")


# Hidden Causal Representation Learning
# GIN
if data_type in ['do not execute']:  # GIN not working --> skipped
    print('---> GIN')
    start_pc = time.time()
    cg_gin, order_gin = GIN(fit_data)
    print('GIN: ' + str((time.time() - start_pc)) + " seconds\n\n")
    execution_times.update({data_type + ", GIN": str(time.time() - start_pc) + "s"})
    pcg_gin = GraphUtils.to_pydot(cg_gin.G, labels=var_names)
    if data_type == 'encoded':
        pcg_gin.write_png(
            graphs_dir + '/onehot/GIN-onehot/encoding_causal_graph_causal-learn_gin.png')
    else:
        pcg_gin.write_png(graphs_dir + '/labelling/GIN-labelling/labelling_causal_graph_causal-learn_gin.png')


# Permutation-based Methods
# GRaSP
score_funcs = ['local_score_BIC']  # local_score_BDeu does not start --> skipped
for sf in score_funcs:
    print('---> GRaSP with score_func = ' + str(sf))
    start_pc = time.time()
    cg_grasp = grasp(fit_data, sf)
    print("GRaSP with score_func = " + str(sf) + ": " + str((time.time() - start_pc)) + " seconds\n\n")
    execution_times.update({data_type + ", GRaSP - " + sf: str(time.time() - start_pc) + "s"})
    pcg_grasp = GraphUtils.to_pydot(cg_grasp, labels=var_names)
    if data_type == 'encoded':
        pcg_grasp.write_png(
            graphs_dir + '/onehot/GRaSP-onehot/encoding_causal_graph_causal-learn_grasp_' + str(sf) + '_.png')
    else:
        pcg_grasp.write_png(
            graphs_dir + '/labelling/GRaSP-labelling/labelling_causal_graph_causal-learn_grasp_' + str(sf) + '_.png')

    grasp_edges = cg_grasp.get_graph_edges()
    np_grasp_edges = []
    for i in range(len(grasp_edges)):
        edge = [int(str(grasp_edges[i].get_node1()).replace('X', ''))-1, int(str(grasp_edges[i].get_node2()).replace('X', ''))-1]
        np_grasp_edges.append(np.array(edge))

    np_grasp_nodes = []
    for edge in np_grasp_edges:
        for node in edge:
            if node not in np_grasp_nodes:
                np_grasp_nodes.append(node)
    np_grasp_names = var_names.to_numpy()
    np_grasp_edges_names = []
    np_grasp_nodes_names = []
    for edge in np_grasp_edges:
        new_edge = []
        for node in edge:
            new_node = var_names[node]
            new_edge.append(new_node)
        np_grasp_edges_names.append(new_edge)
    for node in np_grasp_nodes:
        np_grasp_nodes_names.append(var_names[node])
    np_grasp_edges_names = np.array(np_grasp_edges_names)

    if data_type == 'encoded':
        np.save(npy_dir + '/encoding_causal_graph_causal-learn_grasp_' + str(sf) + '.npy', np_grasp_edges_names)
        with open(txt_dir + '/encoding_causal_graph_causal-learn_grasp_' + str(sf) + '.txt', 'w') as f:
            for edge in np_grasp_edges_names:
                f.write(f"{edge}\n")
    else:
        np.save(npy_dir + '/labelling_causal_graph_causal-learn_grasp_' + str(sf) + '.npy', np_grasp_edges_names)
        with open(txt_dir + '/labelling_causal_graph_causal-learn_grasp_' + str(sf) + '.txt', 'w') as f:
            for edge in np_grasp_edges_names:
                f.write(f"{edge}\n")


# Granger Causality
print('---> Granger')
start_pc = time.time()
G = Granger()
coeff = G.granger_lasso(fit_data)
print('Granger: ' + str((time.time() - start_pc)) + " seconds\n\n")
execution_times.update({data_type + ", Granger - ": str(time.time() - start_pc) + "s"})

num_nodes = coeff.shape[0]
cg_granger = CausalGraph(num_nodes)
for i in range(num_nodes):
    for j in range(num_nodes):
        edge1 = cg_granger.G.get_edge(cg_granger.G.nodes[i], cg_granger.G.nodes[j])
        if edge1 is not None:
            cg_granger.G.remove_edge(edge1)

for i in range(num_nodes):
    for j in range(num_nodes):
        if coeff[i, j] > 0:
            cg_granger.G.add_edge(Edge(cg_granger.G.nodes[i], cg_granger.G.nodes[j], Endpoint.TAIL, Endpoint.ARROW))
        elif coeff[i, j] < 0:
            cg_granger.G.add_edge(Edge(cg_granger.G.nodes[j], cg_granger.G.nodes[i], Endpoint.TAIL, Endpoint.ARROW))

pcg_granger = GraphUtils.to_pydot(cg_granger.G, labels=var_names)
if data_type == 'encoded':
    pcg_granger.write_png(graphs_dir + '/onehot/Granger-onehot/encoding_causal_graph_causal-learn_granger.png')
else:
    pcg_granger.write_png(graphs_dir + '/labelling/Granger-labelling/labelling_causal_graph_causal-learn_granger.png')

np_granger_edges = np.array(cg_granger.find_fully_directed())
np_granger_nodes = []
for edge in np_granger_edges:
    for node in edge:
        if node not in np_granger_nodes:
            np_granger_nodes.append(node)
np_granger_names = var_names.to_numpy()
np_granger_edges_names = []
np_granger_nodes_names = []
for edge in np_granger_edges:
    new_edge = []
    for node in edge:
        new_node = var_names[node]
        new_edge.append(new_node)
    np_granger_edges_names.append(new_edge)
for node in np_granger_nodes:
    np_granger_nodes_names.append(var_names[node])
np_granger_edges_names = np.array(np_granger_edges_names)

if data_type == 'encoded':
    np.save(npy_dir + '/encoding_causal_graph_causal-learn_granger.npy', np_granger_edges_names)
    with open(txt_dir + '/encoding_causal_graph_causal-learn_granger.txt', 'w') as f:
        for edge in np_granger_edges_names:
            f.write(f"{edge}\n")
else:
    np.save(npy_dir + '/labelling_causal_graph_causal-learn_lingam.npy', np_granger_edges_names)
    with open(txt_dir + '/labelling_causal_graph_causal-learn_lingam.txt', 'w') as f:
        for edge in np_granger_edges_names:
            f.write(f"{edge}\n")

if data_type == 'encoded':
    with open('./graphs/causallearn/encoded_execution_times.json', 'w') as f:
        json.dump(execution_times, f)
else:
    with open('./graphs/causallearn/labelled_execution_times.json', 'w') as f:
        json.dump(execution_times, f)
