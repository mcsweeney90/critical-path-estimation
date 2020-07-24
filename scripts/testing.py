#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anything that needs to be tested...
"""

import dill, pathlib, os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
from timeit import default_timer as timer
import sys
sys.path.append('../') 
from Simulator import Platform, HEFT, PEFT, CPOP

####################################################################################################

# Set some parameters for plots.
# See here: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.titlepad'] = 0
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 5
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.

####################################################################################################

# Cholesky.
# nb = 128
# nt = 35
# adt = "perfect_adt"
# with open('nb{}/{}/{}tasks.dill'.format(nb, adt, nt), 'rb') as file:
#     dag = dill.load(file)
# dag.print_info()

# STG
stg_dag_size = 100
stg_dag_path = '../graphs/STG/{}'.format(stg_dag_size)
with open('{}/rand0000.dill'.format(stg_dag_path), 'rb') as file:
    dag = dill.load(file)

plat = Platform(4, name="4P")  
dag.set_costs(plat, target_ccr=1.0, method="related", het_factor=2.0)
dag.print_info()
heft_mkspan = HEFT(dag, plat, cp_type="HEFT")
print("HEFT makespan: {}".format(heft_mkspan))
fulk_mkspan = HEFT(dag, plat, cp_type="Fulkerson")
print("Fulk makespan: {}".format(fulk_mkspan))


# cp_lengths = dag.critical_paths(cp_type="Fulkerson")
# print(cp_lengths)

# single = Platform(8, name="Single_GPU")
# heft_mkspan = HEFT(dag, single)
# print("HEFT makespan: {}".format(heft_mkspan))
# cpop_mkspan = CPOP(dag, single)
# print("CPOP makespan: {}".format(cpop_mkspan))

