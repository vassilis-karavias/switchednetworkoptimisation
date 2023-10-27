# switchednetworkoptimisation
Repository for TF-QKD optimisation


# dependencies
numpy: 1.20.1+\\
graph-tool: 2.37+
pandas: 1.5.2+
scipy: 1.7.3+
cplex: V20.1
matplotlib: 3.6.2+
networkx:2.8.8+
scikit-learn: 1.1.3+
dill:0.3.6+


# How to use:

To carry out the optimisations, it is first necessary to have graph files in the appropriate form for the model. This requires 4 csv files: a hot capacity file, a cold capacity file, a node position file and an edge position file. The files can either be provided as is to the model or you can use the methods provided to generate these files for random graphs.
