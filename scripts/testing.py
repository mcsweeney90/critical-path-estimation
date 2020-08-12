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
from copy import deepcopy
from networkx.drawing.nx_agraph import graphviz_layout
import sys
sys.path.append('../') 
from Simulator import Task, DAG, Platform, HEFT, PEFT, CPOP

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

nb = 128
nt = 35
adt = "perfect_adt"
with open('nb{}/{}/{}tasks.dill'.format(nb, adt, nt), 'rb') as file:
    dag = dill.load(file)
dag.print_info()

# nw, ccr, h, m = 4, 10, 1.0, "UR"
# platform = Platform(nw, name="{}P".format(nw)) 
# with open('{}/rand0085.dill'.format(dag_path), 'rb') as file:
#     dag = dill.load(file)    
    
# for _ in range(100):    
#     dag.set_costs(platform, target_ccr=ccr, method=m, het_factor=h)
#     mkspan = HEFT(dag, platform, cp_type="WF") 
#     print("\nHEFT-WF makespan: {}".format(mkspan))

