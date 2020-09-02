#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing...
"""

import dill
import numpy as np
import networkx as nx
from timeit import default_timer as timer
import sys
sys.path.append('../') 
from Simulator import Task, DAG, Platform, HEFT

sz = 1000
dag_path = '../graphs/STG/{}'.format(sz)
nw, ccr, h, m = 32, 10, 1.0, "UR"
platform = Platform(nw, name="{}P".format(nw)) 
with open('{}/rand0085.dill'.format(dag_path), 'rb') as file:
    dag = dill.load(file)      
    
with open("test.txt", "w") as dest:    
    start = timer() 
    dag.set_costs(platform, target_ccr=ccr, method=m, het_factor=h)
    elapsed = timer() - start
    print("Setting costs took {} minutes".format(elapsed / 60), file=dest)
        
    start = timer()    
    for _ in range(1):    
        mkspan = HEFT(dag, platform) 
    elapsed = timer() - start
    print("HEFT runs took {} minutes".format(elapsed / 60), file=dest)
