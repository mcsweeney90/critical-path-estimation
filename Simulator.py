#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module for implementation of simulator for the task scheduling problem.
"""

import numpy as np
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
from copy import deepcopy
from networkx.drawing.nx_agraph import to_agraph
from statistics import median
from collections import defaultdict

class Task:
    """
    Represents (static) tasks.
    """         
    def __init__(self, task_type=None):
        """
        Create Task object.
        
        Parameters
        ------------------------
        task_type - None/string
        String identifying the name of the task, e.g., "GEMM".
        
        Attributes
        ------------------------
        type - None/string
        Initialized to task_type.
        
        ID - int
        Identification number of the Task in its DAG.
        
        entry - bool
        True if Task has no predecessors, False otherwise.
        
        exit - bool
        True if Task has no successors, False otherwise.
        
        The following 2 attributes are usually set after initialization by functions which
        take a Platform object as a target platform.      
        
        comp_costs - dict
        Dict {Processor ID : cost}        
        
        comm_costs - dict
        Nested dict {(source processor ID, target processor ID) : {child ID : cost}}
        
        The following 3 attributes are set once the task has actually been scheduled.
                
        FT- int/float
        The actual finish time of the Task.
        
        scheduled - bool
        True if Task has been scheduled on a Worker, False otherwise.
        
        where_scheduled - None/int
        The numerical ID of the Worker that the Task has been scheduled on. Often useful.
        
        Comments
        ------------------------
        1. It would perhaps be more useful in general to take all attributes as parameters since this
           is more flexible but as we rarely work at the level of individual Tasks this isn't necessary
           for our purposes.        
        """           
         
        self.type = task_type  
        self.ID = None    
        self.entry = False 
        self.exit = False    
        
        self.comp_costs = {} 
        self.comm_costs = {}     
        
        self.FT = None  
        self.scheduled = False  
        self.where_scheduled = None                 
    
    def reset(self):
        """Resets some attributes to defaults so execution of the task can be simulated again."""
        self.FT = None   
        self.scheduled = False
        self.where_scheduled = None   
        
    def average_cost(self, avg_type="HEFT", weights=None):
        """
        Compute an "average" computational cost for the task. 
        Usually used for setting priorities in HEFT and similar heuristics.
        
        Parameters
        ------------------------
                        
        avg_type - string
        How the average should be computed. 
        Options:
            - "HEFT", use mean values over all processors as in HEFT.
            - "median", use median values over all processors. 
            - "worst", always use largest possible computation cost.
            - "simple worst", always use largest possible computation cost.
            - "best", always use smallest possible computation cost.
            - "simple best", always use smallest possible computation cost.
            - "HEFT-WM", compute mean over all processors, weighted by processing time.
                                         
        Returns
        ------------------------
        float 
        The average computation cost of the Task. 
        
        Notes
        ------------------------
        1. "median", "worst", "simple worst", "best", "simple best" were all considered by Zhao and Sakellariou (2003). 
        """            
        
        if avg_type == "HEFT" or avg_type == "mean" or avg_type == "MEAN" or avg_type == "M":
            return sum(v for v in self.comp_costs.values()) / len(self.comp_costs)
        elif avg_type == "median" or avg_type == "MEDIAN":
            return median(self.comp_costs.values())
        elif avg_type == "worst" or avg_type == "W" or avg_type == "simple worst" or avg_type == "SW":
            return max(self.comp_costs.values())
        elif avg_type == "best" or avg_type == "B" or avg_type == "simple best" or avg_type == "sb":
            return min(self.comp_costs.values())   
        elif avg_type == "HEFT-WM" or avg_type == "WM":
            if weights is not None:              
                return sum(weights[k]*v for k, v in self.comp_costs.items()) / sum(weights.values())
            s = sum(1/v for v in self.comp_costs.values())
            return len(self.comp_costs) / s         
        raise ValueError('Unrecognized avg_type specified for average_cost.') 
        
    def average_comm_cost(self, child, avg_type="HEFT", weights=None):
        """
        Compute an "average" communication cost between Task and one of its children. 
        Usually used for setting priorities in HEFT and similar heuristics.
        
        Parameters
        ------------------------
                        
        avg_type - string
        How the average should be computed. 
        Options:
            - "HEFT", use mean values over all processors as in HEFT.
            - "median", use median values over all processors. 
            - "worst", always use largest possible computation cost.
            - "simple worst", always use largest possible computation cost.
            - "best", always use smallest possible computation cost.
            - "simple best", always use smallest possible computation cost.
            - "HEFT-WM", compute mean over all processors, weighted by processing time.
                                         
        Returns
        ------------------------
        float 
        The average communication cost between the Task and the child. 
        
        Notes
        ------------------------
        1. "median", "worst", "simple worst", "best", "simple best" were all considered by Zhao and Sakellariou (2003). 
        """         
                
        if avg_type == "HEFT" or avg_type == "mean" or avg_type == "MEAN" or avg_type == "M":
            return sum(v for v in self.comm_costs[child.ID].values()) / len(self.comm_costs[child.ID])            
        elif avg_type == "median" or avg_type == "MEDIAN":
            return median(self.comm_costs[child.ID].values())
        elif avg_type == "worst" or avg_type == "W":
            task_worst = max(self.comp_costs, key=self.comp_costs.get)
            child_worst = max(child.comp_costs, key=child.comp_costs.get)            
            return self.comm_costs[child.ID][(task_worst, child_worst)]
        elif avg_type == "simple worst" or avg_type == "SW":
            return max(self.comm_costs[child.ID].values())
        elif avg_type == "best" or avg_type == "B":
            task_best = min(self.comp_costs, key=self.comp_costs.get)
            child_best = min(child.comp_costs, key=child.comp_costs.get)            
            return self.comm_costs[child.ID][(task_best, child_best)]
        elif avg_type == "simple best" or avg_type == "sb":
            return 0.0 # min(v for k, v in self.comm_costs[child.ID].items() if k[0] != k[1])   
        elif avg_type == "HEFT-WM" or avg_type == "WM":
            s1 = sum(1/v for v in self.comp_costs.values())
            s2 = sum(1/v for v in child.comp_costs.values())
            cbar = 0.0
            for k, v in self.comm_costs[child.ID].items():
                t_w = self.comp_costs[k[0]]
                c_w = child.comp_costs[k[1]]                
                cbar += v/(t_w * c_w) 
            cbar /= (s1 * s2)
            return cbar                    
        raise ValueError('Unrecognized avg_type specified for average_comm_cost.')  
        
    def aoa_edge_cdf(self, child, weighted=False):
        """
        Helper function for computing Fulkerson numbers in the DAG critical_paths method.
        Basically returns the CDF of the edge from Task to child in an activity-on-arc (AoA)
        version of the graph.
        If weighted, weights the pmfs according to computation costs.
        """   

        if weighted:
            s1 = sum(1/v for v in self.comp_costs.values())
            s2 = sum(1/v for v in child.comp_costs.values())
                   
        pmf = {}
        for k, v in self.comm_costs[child.ID].items():
            w = v + self.comp_costs[k[0]] 
            if child.exit:
                w += child.comp_costs[k[1]]
            if weighted:
                t_w = self.comp_costs[k[0]]
                c_w = child.comp_costs[k[1]]                
                pr = 1/(t_w * c_w)
                pr /= (s1 * s2)
            else:
                pr = 1
            try:
                pmf[w] += pr
            except KeyError:
                pmf[w] = pr
                    
        if not weighted:
            d = len(self.comm_costs[child.ID])                    
        cdf, x = {}, 0.0
        for k, v in sorted(pmf.items(), key=lambda item: item[0]):
            cdf[k] = v/d + x if not weighted else v + x
            x = cdf[k]
        return cdf

class DAG:
    """
    Represents a task DAG.   
    """
    def __init__(self, G, name=None): 
        """
        The DAG is a collection of Tasks with a topology defined by a Networkx DiGraph object.        
        
        Parameters
        ------------------------        
        G - DiGraph
        A Networkx DiGraph with Task nodes giving the topology of the DAG.
        
        name - string
        The name of the application the DAG represents, e.g., "Cholesky".
        
        Attributes
        ------------------------
        name - string
        Ditto above.
        
        graph - DiGraph from Networkx module
        Represents the topology of the DAG.
        
        n_tasks - int
        The number of tasks in the DAG.
               
        n_edges - int
        The number of edges in the DAG.    

        top_sort - list of Tasks
        List of the tasks in a topologically-sorted order. Often useful.

        costs_set - bool
        True if costs have been set, False otherwise.

        target_platform - string
        Name of the target platform that costs represent.                  
        """  
        
        self.name = name 
        self.graph = G
        self.n_tasks = len(G)  
        self.n_edges = G.number_of_edges()   
        self.top_sort = list(nx.topological_sort(self.graph)) 
        # Give all tasks an ID in topological order.
        for i, t in enumerate(self.top_sort):
            t.ID = i
            if not list(self.graph.predecessors(t)):
                t.entry = True
            elif not list(self.graph.successors(t)):
                t.exit = True
        self.costs_set = False
        self.target_platform = None
        
    def reset(self):
        """Resets some Task attributes to defaults so scheduling of the DAG can be simulated again."""
        for task in self.graph:
            task.reset() 

    def scheduled(self):
        """Returns True all the tasks in the DAG have been scheduled, False if not."""
        return all(task.scheduled for task in self.graph)   
    
    def ready_to_schedule(self, task):
        """
        Determine if task is ready to schedule - i.e., all precedence constraints have been 
        satisfied or it is an entry task.
        
        Parameters
        ------------------------
        task - Task object
        Task in the DAG.               
                                         
        Returns
        ------------------------
        bool
        True if Task can be scheduled, False otherwise.         
        
        Notes
        ------------------------
        1. Returns False if Task has already been scheduled.
        """
        
        if task.scheduled:
            return False  
        if task.entry: 
            return True
        for parent in self.graph.predecessors(task):
            if not parent.scheduled:
                return False
        return True    
    
    def get_ready_tasks(self):
        """
        Identify the tasks that are ready to schedule.               

        Returns
        ------------------------                          
        List
        All tasks in the DAG that are ready to be scheduled.                 
        """       
        return list(t for t in self.graph if self.ready_to_schedule(t))
    
    def makespan(self, partial=False):
        """
        Compute the makespan of the DAG.
        
        Parameters
        ------------------------        
        partial - bool
        If True, only computes makespan of all tasks that have been scheduled so far, not the entire DAG. 

        Returns
        ------------------------         
        int/float
        The makespan of the (possibly incomplete) DAG.        
        """         
        if partial:
            return max(t.FT for t in self.graph if t.FT is not None)  
        return max(t.FT for t in self.graph if t.exit) 
    
    def set_costs(self, platform, target_ccr=1.0, method="R", het_factor=1.0):
        """
        Sets computation and communication costs for randomly generated DAGs (e.g., from the STG).
        Notes:
            - het_factor in the interval (0, 2).
        """   
        
        if method == "unrelated" or method == "UR":        
            # Set computation costs.
            avg_task_cost = np.random.uniform(1, 100)
            for task in self.top_sort:
                wbar = np.random.uniform(0, 2 * avg_task_cost)
                for w in platform.workers:
                    task.comp_costs[w.ID] = np.random.uniform(wbar * (1 - het_factor/2), wbar * (1 + het_factor/2))                
            # Set communication costs.
            total_edge_costs = (avg_task_cost * self.n_tasks) / target_ccr
            avg_edge_cost = total_edge_costs / self.n_edges 
            for task in self.top_sort:
                for child in self.graph.successors(task):
                    task.comm_costs[child.ID] = {}
                    wbar = np.random.uniform(0, 2 * avg_edge_cost)
                    wbar *= (1 + platform.n_workers)/(platform.n_workers) # Adjust for zero costs from processor to themselves.
                    for u in platform.workers:
                        for v in platform.workers:
                            c = 0.0 if u == v else np.random.uniform(wbar * (1 - het_factor/2), wbar * (1 + het_factor/2))
                            task.comm_costs[child.ID][(u.ID, v.ID)] = c
        elif method == "related" or method == "R":
            avg_power = np.random.uniform(1, 100)
            powers = {}
            for w in platform.workers:
                powers[w.ID] = np.random.uniform(avg_power * (1 - het_factor/2), avg_power * (1 + het_factor/2))
            # Set computation costs.
            avg_task_cost = np.random.uniform(1, 100)
            for task in self.top_sort:
                t = np.random.uniform(0, 2 * avg_task_cost)
                for w in platform.workers:
                    task.comp_costs[w.ID] = t * np.random.gamma(shape=1.0, scale=powers[w.ID])
            # Set communication costs.
            total_edge_costs = sum(t.average_cost() for t in self.top_sort) / target_ccr
            avg_edge_cost = total_edge_costs / self.n_edges 
            for task in self.top_sort:
                for child in self.graph.successors(task):
                    task.comm_costs[child.ID] = {}
                    wbar = np.random.uniform(0, 2 * avg_edge_cost)
                    wbar *= (1 + platform.n_workers)/(platform.n_workers) # Adjust for zero costs from processor to themselves.
                    for u in platform.workers:
                        for v in platform.workers:
                            c = 0.0 if u == v else np.random.uniform(wbar * (1 - het_factor/2), wbar * (1 + het_factor/2))
                            task.comm_costs[child.ID][(u.ID, v.ID)] = c      
        elif method == "diagram":
            for task in self.top_sort:
                for w in platform.workers:
                    task.comp_costs[w.ID] = np.random.randint(1, 10)
                for child in self.graph.successors(task):
                    task.comm_costs[child.ID] = {}
                    for u in platform.workers:
                        for v in platform.workers:
                            c = 0 if u == v else np.random.randint(1, 10)
                            task.comm_costs[child.ID][(u.ID, v.ID)] = c
        # Set DAG attributes.                  
        self.costs_set = True
        self.target_platform = platform.name
                  
                
    def minimal_serial_time(self):
        """
        Computes the minimum makespan of the DAG on a single Worker.
        
        Returns
        ------------------------                          
        float
        The minimal serial time.      
        
        Notes
        ------------------------                          
        1. Assumes all task computation costs are set so no need to take Platform object as input.        
        """ 
        
        workers = list(k for k in self.top_sort[0].comp_costs)    # Assumes all workers can execute all tasks...    
        worker_serial_times = list(sum(t.comp_costs[w] for t in self.graph) for w in workers)        
        return min(worker_serial_times)
    
    def CCR(self, avg_type="HEFT"):
        """
        Compute and set the computation-to-communication ratio (CCR) for the DAG.
        Assumes all costs set.    
        """        
        exp_comm, exp_comp = 0.0, 0.0
        for task in self.top_sort:
            exp_comp += task.average_cost(avg_type=avg_type)
            children = self.graph.successors(task)
            for child in children:
                exp_comm += task.average_comm_cost(child, avg_type=avg_type)
        return exp_comp / exp_comm
    
    def optimistic_cost_table(self):
        """
        Optimistic Cost Table as used in PEFT heuristic. 
        Note this is slightly different to the (upward, optimistic) conditional critical path below. 
        """  
        
        workers = list(k for k in self.top_sort[0].comp_costs)  
        d = {}
        for w in workers:
            for v in workers:
                d[(w, v)] = 0.0 if w == v else 1.0   
        OCT = {}        
        backward_traversal = list(reversed(self.top_sort))
        for task in backward_traversal:
            OCT[task.ID] = {}
            for w in workers:
                OCT[task.ID][w] = 0.0
                if task.exit:
                    continue
                child_values = []
                for child in self.graph.successors(task):
                    action_values = [OCT[child.ID][v] + d[(w, v)] * task.average_comm_cost(child) + child.comp_costs[v] for v in workers]
                    child_values.append(min(action_values))
                OCT[task.ID][w] += max(child_values)      
        return OCT    
    
    def critical_paths(self, direction="upward", cp_type="HEFT", avg_type="HEFT", mc_samples=1000):
        """
        Compute critical path length estimates to/from all tasks.
        
        Parameters
        ------------------------
        
        direction - string in ["upward", "downward"]
        Whether to compute critical path length from root to task (downward) or from task to leaves (upward).
        
        cp_type - string 
        What method to use for estimating the critical path lengths, e.g., "HEFT", which is to use upward ranks.
        
        avg_type - string
        Only used if cp_type == "HEFT".
        How the tasks and edges should be weighted in platform.average_comm_cost and task.average_execution_cost.
        Default is "HEFT" which is mean values over all processors. See referenced methods for more options.
        
        mc_samples - string
        Only used if cp_type == "MC".
        How many Monte Carlo realizations/samples to use.
             

        Returns
        ------------------------                          
        cp_lengths - dict
        Scheduling list of all Task objects.               
        """    
        
        cp_lengths = {}
        
        if cp_type == "optimistic" or cp_type == "pessimistic" or cp_type == "LB":
            CCP = self.conditional_critical_paths(direction, cp_type)
            for t in self.top_sort:
                if cp_type == "optimistic" or cp_type == "LB":
                    cp_lengths[t] = min(CCP[t.ID].values())
                elif cp_type == "pessimistic":
                    cp_lengths[t] = max(CCP[t.ID].values())
        elif cp_type == "HEFT" or cp_type == "H" or cp_type == "W":        
            if direction == "upward": 
                backward_traversal = list(reversed(self.top_sort))  
                for t in backward_traversal:
                    cp_lengths[t] = t.average_cost(avg_type=avg_type)
                    try:
                        cp_lengths[t] += max(t.average_comm_cost(s, avg_type=avg_type) + cp_lengths[s] for s in self.graph.successors(t))
                    except ValueError:
                        pass
            elif direction == "downward":
                for t in self.top_sort:
                    cp_lengths[t] = 0.0
                    try:
                        cp_lengths[t] += max(p.average_cost(avg_type=avg_type) + p.average_comm_cost(t, avg_type=avg_type) +
                                  cp_lengths[p] for p in self.graph.predecessors(t))
                    except ValueError:
                        pass    
        elif cp_type == "Fulkerson" or cp_type == "F" or cp_type == "WF":
            fulk_weight = True if cp_type == "WF" else False
            if direction == "upward": 
                backward_traversal = list(reversed(self.top_sort))
                for t in backward_traversal:
                    if t.exit:
                        cp_lengths[t] = 0.0    
                        continue
                    children = list(self.graph.successors(t))                  
                    # Find alpha and the potential z values to check.
                    edge_cdfs = {}
                    alpha, Z = 0.0, []
                    for c in children: 
                        cdf = t.aoa_edge_cdf(c, weighted=fulk_weight)
                        Z += list(cp_lengths[c] + v for v in cdf)
                        alpha = max(alpha, min(cdf))  
                        edge_cdfs[c.ID] = cdf 
                    # Compute f. 
                    cp_lengths[t] = 0.0
                    Z = list(set(Z))    # TODO: might still need a check to prevent rounding errors.
                    for z in Z:
                        if alpha - z > 1e-6:   
                            continue
                        # Iterate over edges and compute the two products.
                        plus, minus = 1, 1                
                        for c in children:
                            zdash = z - cp_lengths[c] 
                            p, m = 0.0, 0.0
                            for k, v in edge_cdfs[c.ID].items():
                                if abs(zdash - k) < 1e-6:
                                    p = v
                                    break
                                elif k - zdash > 1e-6:
                                    break
                                else:
                                    p, m = v, v 
                            minus *= m 
                            plus *= p
                        # Add to f.                                    
                        cp_lengths[t] += z * (plus - minus)
            elif direction == "downward": # TODO: check this at some point but don't actually use anywhere so not a priority.
                for t in self.top_sort:
                    if t.entry:
                        cp_lengths[t] = 0.0
                        continue
                    parents = list(self.graph.predecessors(t)) 
                    # Find alpha and the potential z values to check.
                    edge_cdfs = {}
                    alpha, Z = 0.0, []
                    for p in parents: 
                        cdf = p.aoa_edge_cdf(t, weighted=fulk_weight)
                        Z += list(cp_lengths[p] + v for v in cdf)
                        alpha = max(alpha, min(cdf))  
                        edge_cdfs[p.ID] = cdf 
                    # Compute f. 
                    cp_lengths[t] = 0.0
                    Z = list(set(Z))    # TODO: might still need a check to prevent rounding errors.
                    for z in Z:
                        if alpha - z > 1e-6:   
                            continue
                        # Iterate over edges and compute the two products.
                        plus, minus = 1, 1                
                        for q in parents:
                            zdash = z - cp_lengths[q] 
                            p, m = 0.0, 0.0
                            for k, v in edge_cdfs[q.ID].items():
                                if abs(zdash - k) < 1e-6:
                                    p = v
                                    break
                                elif k - zdash > 1e-6:
                                    break
                                else:
                                    p, m = v, v  
                            minus *= m 
                            plus *= p
                        # Add to f.                                    
                        cp_lengths[t] += z * (plus - minus)
                        
        elif cp_type == "Monte Carlo" or cp_type == "MC" or cp_type == "WMC":
            workers = list(k for k in self.top_sort[0].comp_costs)
            for _ in range(mc_samples):
                # Generate an assignment.
                assignment = {}
                for t in self.top_sort:
                    if cp_type == "WMC":
                        s = sum(1/v for v in t.comp_costs.values())
                        p = list((1/v)/s for v in t.comp_costs.values())
                        assignment[t.ID] = np.random.choice(workers, p=p)
                    else:                        
                        assignment[t.ID] = np.random.choice(workers)
                    for s in self.graph.successors(t):
                        assignment[(t.ID, s.ID)] = np.random.choice(list(t.comm_costs[s.ID].values()))
                # Compute the critical path lengths.
                fixed_lengths = {}
                if direction == "upward":
                    backward_traversal = list(reversed(self.top_sort)) 
                    for t in backward_traversal:
                        fixed_lengths[t.ID] = t.comp_costs[assignment[t.ID]]
                        try:
                            # fixed_lengths[t.ID] += max(t.comm_costs[s.ID][(assignment[t.ID], assignment[s.ID])] + fixed_lengths[s.ID] for s in self.graph.successors(t))
                            fixed_lengths[t.ID] += max(assignment[(t.ID, s.ID)] + fixed_lengths[s.ID] for s in self.graph.successors(t))
                        except ValueError:
                            pass
                        try:
                            cp_lengths[t] += fixed_lengths[t.ID]
                        except KeyError:
                            cp_lengths[t] = fixed_lengths[t.ID]
                elif direction == "downward":
                    for t in self.top_sort:
                        fixed_lengths[t.ID] = 0.0
                        try:
                            fixed_lengths[t.ID] += max(p.comp_costs[assignment[p.ID]] + p.comm_costs[t.ID][(assignment[p.ID], assignment[t.ID])] +
                                      fixed_lengths[p.ID] for p in self.graph.predecessors(t))
                        except ValueError:
                            pass
                        try:
                            cp_lengths[t] += fixed_lengths[t.ID]
                        except KeyError:
                            cp_lengths[t] = fixed_lengths[t.ID]
            for t in self.top_sort:
                cp_lengths[t] /= mc_samples # Not really necessary...           
                    
        return cp_lengths 
    
    def conditional_critical_paths(self, direction="upward", cp_type="optimistic", lookahead=False):
        """
        Computes critical path estimates, either upward or downward, of all tasks assuming they are 
        scheduled on a given processor.
        
        Parameters
        ------------------------
        direction - string
        What direction we work in.
        
        cp_type - string
        What method to use for estimating the critical path lengths, e.g., "HEFT", which is to use upward ranks.
        
        lookahead - bool
        If True, don't include current task (as when using conditional critical paths for processor selection).

        Returns
        ------------------------                          
        CCP - Nested dict
        Conditional critical path estimates in the form {Task 1: {Worker 1 : c1, Worker 2 : c2, ...}, ...}.         
        
        Notes
        ------------------------ 
        1. No target platform is necessary since costs assumed to be set.
        """  
        
        workers = list(k for k in self.top_sort[0].comp_costs) 
        CCP = {}          
        if direction == "upward":
            backward_traversal = list(reversed(self.top_sort))
            for task in backward_traversal:
                CCP[task.ID] = {}
                for w in workers:
                    CCP[task.ID][w] = task.comp_costs[w] if not lookahead else 0.0
                    if task.exit:
                        continue
                    child_values = []
                    for child in self.graph.successors(task):
                        if lookahead:
                            action_values = [CCP[child.ID][v] + task.comm_costs[child.ID][(w, v)] + child.comp_costs[v] for v in workers]
                        else:
                            action_values = [CCP[child.ID][v] + task.comm_costs[child.ID][(w, v)] for v in workers]
                            
                        if cp_type == "optimistic" or cp_type == "LB":
                            child_values.append(min(action_values))                            
                        elif cp_type == "pessimistic":
                            child_values.append(max(action_values))
                        elif cp_type == "M":
                            child_values.append(sum(action_values) / len(action_values))
                        elif cp_type == "WM":
                            sk = sum(1/(child.comp_costs[w] + CCP[child.ID][w]) for w in workers)
                            child_values.append(sum(a/(child.comp_costs[w] + CCP[child.ID][w]) for a, w in zip(action_values, workers)) / sk)
                    CCP[task.ID][w] += max(child_values)             
        else: # TODO: M and WM.
            for task in self.top_sort:
                CCP[task.ID] = {}
                for w in workers:
                    CCP[task.ID][w] = task.comp_costs[w]
                    if task.entry:
                        continue
                    parent_values = []
                    for parent in self.graph.predecessors(task):
                        action_values = [CCP[parent.ID][v] + parent.comm_costs[task.ID][(v, w)] for v in workers]
                        if cp_type == "optimistic" or cp_type == "LB":
                            parent_values.append(min(action_values))
                        elif cp_type == "pessimistic":
                            parent_values.append(max(action_values))
                    CCP[task.ID][w] += max(parent_values)   
        return CCP        
    
    def critical_path_priorities(self, direction="upward", cp_type="HEFT", avg_type="HEFT", mc_samples=1000, return_ranks=False):
        """
        Sorts all tasks in the DAG according to the estimated critical path lengths upward/downward.
        
        Parameters
        ------------------------
        direction - string
        What direction we work in.
        
        cp_type - string
        What method to use for estimating the critical path lengths, e.g., "HEFT", which is to use upward ranks.
        
        avg_type - string
        Only used if cp_type == "HEFT".
        How the tasks and edges should be weighted in platform.average_comm_cost and task.average_execution_cost.
        Default is "HEFT" which is mean values over all processors. See referenced methods for more options.
        
        mc_samples - string
        Only used if cp_type == "MC".
        How many Monte Carlo realizations/samples to use.
        
        return_ranks - bool
        If True, method also returns the rank values for all tasks.
        
        Returns
        ------------------------                          
        priority_list - list
        Scheduling list of all Task objects.
        
        If return_ranks == True:
        task_ranks - dict
        Gives the actual ranks of all tasks in the form {task : rank}.               
        """      
        task_ranks = self.critical_paths(direction, cp_type, avg_type, mc_samples)
        if direction == "upward":
            priority_list = list(sorted(task_ranks, key=task_ranks.get, reverse=True))
        else:
            priority_list = list(sorted(task_ranks, key=task_ranks.get))
        if return_ranks:
            return priority_list, task_ranks
        return priority_list  
    
    def conditional_critical_path_priorities(self, platform, CCP=None, cp_type="WM", rank_avg="WM", return_ranks=False):
        """
        TODO.              
        """      
        
        if CCP is None:  
            CCP = self.conditional_critical_paths(cp_type=cp_type, lookahead=True)        
        
        task_ranks = {}        
        backward_traversal = list(reversed(self.top_sort)) 
        for t in backward_traversal:        
            if rank_avg == "MEAN" or rank_avg == "M":
                task_ranks[t] = sum(t.comp_costs[w.ID] + CCP[t.ID][w.ID] for w in platform.workers) / platform.n_workers  
            elif rank_avg == "WEIGHTED MEAN" or rank_avg == "WM":
                s = sum(1/(t.comp_costs[w.ID] + CCP[t.ID][w.ID]) for w in platform.workers)
                task_ranks[t] = platform.n_workers / s
            try:
                task_ranks[t] += max(task_ranks[s] for s in self.graph.successors(t))
            except ValueError:
                pass    
        # Sort task ranks into priority list.
        priority_list = list(sorted(task_ranks, key=task_ranks.get, reverse=True)) 
        
        if return_ranks:
            return priority_list, task_ranks
        return priority_list 
                     
    def draw_graph(self):
        """
        Draws the DAG.        

        Notes
        ------------------------                           
        1. See https://stackoverflow.com/questions/39657395/how-to-draw-properly-networkx-graphs       
        """        
        G = deepcopy(self.graph)        
        G.graph['graph'] = {'rankdir':'TD'}  
        G.graph['node']={'shape':'circle', 'color':'#348ABD', 'style':'filled', 'fillcolor':'#E5E5E5', 'penwidth':'3.0'}
        G.graph['edges']={'arrowsize':'4.0', 'penwidth':'5.0'}       
        A = to_agraph(G)        
        # Add identifying colors if task types are known.
        for task in G:
            if task.type == "GEMM":
                n = A.get_node(task)  
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#E24A33'
                n.attr['label'] = 'G'
            elif task.type == "POTRF":
                n = A.get_node(task)   
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#348ABD'
                n.attr['label'] = 'P'
            elif task.type == "SYRK":
                n = A.get_node(task)   
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#988ED5'
                n.attr['label'] = 'S'
            elif task.type == "TRSM":
                n = A.get_node(task)    
                n.attr['color'] = 'black'
                n.attr['fillcolor'] = '#FBC15E'
                n.attr['label'] = 'T' 
        A.layout('dot')
        A.draw('{}.png'.format(self.name)) 
    
    def print_info(self, return_mst_and_cp=False, detailed=False, filepath=None):
        """
        Print basic information about the DAG, either to screen or as txt file.
        
        Parameters
        ------------------------
        return_mst_and_cp - bool
        If True, return the MST and CP/LB (sometimes prevents computing them again.)
        
        detailed - bool
        If True, print information about individual Tasks.
        
        filepath - string
        Destination for txt file.                           
        """
        
        print("--------------------------------------------------------", file=filepath)
        print("DAG INFO", file=filepath)
        print("--------------------------------------------------------", file=filepath)   
        print("Name: {}".format(self.name), file=filepath)
        
        # Basic topological information.
        print("Number of tasks: {}".format(self.n_tasks), file=filepath)
        print("Number of edges: {}".format(self.n_edges), file=filepath)
        max_edges = (self.n_tasks * (self.n_tasks - 1)) / 2 
        edge_density = self.n_edges / max_edges 
        print("Edge density: {}".format(edge_density), file=filepath)                
            
        if self.costs_set:  
            print("--------------------------------------------------------", file=filepath) 
            print("Target platform: {}".format(self.target_platform), file=filepath)
            mst = self.minimal_serial_time()
            print("Minimal serial time: {}".format(mst), file=filepath)
            OCP = self.conditional_critical_paths(direction="downward", cp_type="optimistic")
            cp = max(min(OCP[task.ID][p] for p in OCP[task.ID]) for task in self.graph if task.exit) 
            print("Optimal critical path length: {}".format(cp), file=filepath) 
            ccr = self.CCR()
            print("Computation-to-communication ratio: {}".format(ccr), file=filepath)        
                    
        if detailed:
            print("\n--------------------------------------------------------", file=filepath) 
            print("DETAILED BREAKDOWN OF TASKS IN DAG:", file=filepath)
            print("--------------------------------------------------------", file=filepath) 
            for task in self.graph:
                print("\nTask ID: {}".format(task.ID), file=filepath)
                if task.entry:
                    print("Entry task.", file=filepath)
                if task.exit:
                    print("Exit task.", file=filepath)
                if task.type is not None:
                    print("Task type: {}".format(task.type), file=filepath)              
        print("--------------------------------------------------------", file=filepath) 
        
        if return_mst_and_cp:
            return mst, cp
          
class Worker:
    """
    Represents a processor. 
    """
    def __init__(self, ID=None):
        """
        Create the Worker object.
        
        Parameters
        --------------------        
        ID - Int
        Assigns an integer ID to the task. Often very useful.        
        """        
        
        self.ID = ID   
        self.load = []  # Tasks scheduled on the processor.
        self.idle = True    # True if no tasks currently scheduled on the processor. 
                
    def earliest_finish_time(self, task, dag, platform, insertion=True):
        """
        Returns the estimated earliest start time for a Task on the Worker.
        
        Parameters
        ------------------------
        task - Task object 
        Represents a (static) task.
        
        dag - DAG object 
        The DAG to which the task belongs.
              
        platform - Platform object
        The Node object to which the Worker belongs.
        Needed for calculating communication costs.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.
        
        Returns
        ------------------------
        float 
        The earliest finish time for task on Worker.        
        """    
        
        task_cost = task.comp_costs[self.ID] 
        
        # If no tasks scheduled on processor...
        if self.idle:   
            if task.entry: 
                return (task_cost, 0)
            else:
                return (task_cost + max(p.FT + p.comm_costs[task.ID][(p.where_scheduled, self.ID)] for p in dag.graph.predecessors(task)), 0)                  
            
        # At least one task already scheduled on processor... 
                
        # Find earliest time all task predecessors have finished and the task can theoretically start.     
        drt = 0
        if not task.entry:                    
            parents = dag.graph.predecessors(task) 
            drt += max(p.FT + p.comm_costs[task.ID][(p.where_scheduled, self.ID)] for p in parents)  
        
        if not insertion:
            return (task_cost + max(self.load[-1][2], drt), -1)
        
        # Check if it can be scheduled before any other task in the load.
        prev_finish_time = 0.0
        for i, t in enumerate(self.load):
            if t[1] < drt:
                prev_finish_time = t[2]
                continue
            poss_finish_time = max(prev_finish_time, drt) + task_cost
            if poss_finish_time <= t[1]:
                return (poss_finish_time, i) 
            prev_finish_time = t[2]
        
        # No valid gap found.
        return (task_cost + max(self.load[-1][2], drt), -1)    
        
    def schedule_task(self, task, finish_time=None, load_idx=None, dag=None, platform=None, insertion=True):
        """
        Schedules the task on the Worker.
        
        Parameters
        ------------------------
        task - Task object (see Graph.py module)
        Represents a (static) task.
                
        dag - DAG object (see Graph.py module)
        The DAG to which the task belongs.
              
        platform - Platform object
        The Platform object to which the Worker belongs. 
        Needed for calculating communication costs.
        
        insertion - bool
        If True, use insertion-based scheduling policy - i.e., task can be scheduled 
        between two already scheduled tasks, if permitted by dependencies.                        
        """         
                        
        # Set task attributes.
        if finish_time is None:
            finish_time, load_idx = self.earliest_finish_time(task, dag, platform, insertion=insertion) 
        
        start_time = finish_time - task.comp_costs[self.ID] 
        
        # Add to load.           
        if self.idle or not insertion or load_idx < 0:             
            self.load.append((task.ID, start_time, finish_time, task.type))  
            if self.idle:
                self.idle = False
        else: 
            self.load.insert(load_idx, (task.ID, start_time, finish_time, task.type))                
        
        # Set the task attributes.
        task.FT = finish_time 
        task.scheduled = True
        task.where_scheduled = self.ID  

    def unschedule_task(self, task):
        """
        Unschedules the task on the Worker.
        
        Parameters
        ------------------------
        task - Task object
        Represents a (static) task.                 
        """
        
        # Remove task from the load.
        for t in self.load:
            if t[0] == task.ID:
                self.load.remove(t)
                break
        # Revert Worker to idle if necessary.
        if not len(self.load):
            self.idle = True
        # Reset the task itself.    
        task.reset()                         
        
    def print_schedule(self, filepath=None):
        """
        Print the current tasks scheduled on the Worker, either to screen or as txt file.
        
        Parameters
        ------------------------
        filepath - string
        Destination for schedule txt file.                           
        """        
        print("WORKER {}: ".format(self.ID), file=filepath)
        for t in self.load:
            type_info = " Task type: {},".format(t[3]) if t[3] is not None else ""
            print("Task ID: {},{} Start time = {}, finish time = {}.".format(t[0], type_info, t[1], t[2]), file=filepath)  

class Platform:
    """          
    A Platform is basically just a collection of Worker objects.
    """
    def __init__(self, n_workers, name=None):
        """
        Initialize the Node by giving the number of processors/workers.
        
        Parameters
        ------------------------
        n_workers - int
        Number of workers.
        
        name - string
        An identifying name for the platform. Often useful.
        """
        
        self.name = name     
        self.n_workers = n_workers      # Often useful.
        self.workers = [Worker(ID="P{}".format(i)) for i in range(self.n_workers)]       # List of all Worker objects.       
    
    def reset(self):
        """Resets some attributes to defaults so we can simulate the execution of another DAG."""
        for w in self.workers:
            w.load = []   
            w.idle = True                    
    
    def print_info(self, print_schedule=False, filepath=None):
        """
        Print basic information about the Platform, either to screen or as txt file.
        
        Parameters
        ------------------------
        filepath - string
        Destination for txt file.                           
        """        
        print("----------------------------------------------------------------------------------------------------------------", file=filepath)
        print("PLATFORM INFO", file=filepath)
        print("----------------------------------------------------------------------------------------------------------------", file=filepath)
        print("Name: {}".format(self.name), file=filepath)
        print("{} workers".format(self.n_workers), file=filepath)
        print("----------------------------------------------------------------------------------------------------------------\n", file=filepath)  
        
        if print_schedule:
            print("----------------------------------------------------------------------------------------------------------------", file=filepath)
            print("CURRENT SCHEDULE", file=filepath)
            print("----------------------------------------------------------------------------------------------------------------", file=filepath)
            for w in self.workers:
                w.print_schedule(filepath=filepath)  
            mkspan = max(w.load[-1][2] for w in self.workers if w.load) 
            print("\nMAKESPAN: {}".format(mkspan), file=filepath)            
            print("----------------------------------------------------------------------------------------------------------------\n", file=filepath)
    
    def follow_schedule(self, dag, schedule):
        """
        Follow the input schedule.
        """        
        
        info = {}     
        # Compute makespan.
        for task in schedule:
            p = schedule[task]
            self.workers[p].schedule_task(task, dag=dag, platform=self)  
        mkspan = dag.makespan() 
        info["MAKESPAN"] = mkspan        
        
        # Reset DAG and platform.
        dag.reset()
        self.reset() 
        
        # Compute CCR and critical path of the fixed-cost DAG.
        backward_traversal = list(reversed(dag.top_sort))  
        cp_lengths, total_comp, total_comm = {}, 0.0, 0.0
        for task in backward_traversal:            
            w_t = task.comp_costs[schedule[task]] 
            total_comp += w_t
            cp_lengths[task.ID] = w_t 
            children = list(dag.graph.successors(task))
            maximand = 0.0
            source = schedule[task]
            for c in children:
                target = schedule[c]
                edge_cost = task.comm_costs[c.ID][(source, target)] 
                total_comm += edge_cost
                maximand = max(maximand, edge_cost + cp_lengths[c.ID])
            cp_lengths[task.ID] += maximand
        
        ccr = total_comp / total_comm
        info["CCR"] = ccr        
        cp = cp_lengths[dag.top_sort[0].ID]
        info["CRITICAL PATH"] = cp
        slr = mkspan / cp
        info["SCHEDULE LENGTH RATIO"] = slr
        
        return info             
            
# =============================================================================
# Heuristics.   
# =============================================================================            
            
def HEFT(dag, platform, priority_list=None, cp_type="HEFT", avg_type="HEFT", 
         return_schedule=False, schedule_dest=None, schedule_img_dest=None, img_name=None):
    """
    Heterogeneous Earliest Finish Time.
    'Performance-effective and low-complexity task scheduling for heterogeneous computing',
    Topcuoglu, Hariri and Wu, 2002.
    
    Parameters
    ------------------------    
    dag - DAG object 
    Represents the task DAG to be scheduled.
          
    platform - Platform object 
    Represents the target platform.  
    
    priority_list - None/list
    If not None, an ordered list which gives the order in which tasks are to be scheduled. 
    
    avg_type - string
    How the tasks and edges should be weighted in dag.sort_by_upward_rank.
    Default is "HEFT" which is mean values over all processors as in the original paper. 
    See task.average_comm_cost and task.average_cost for other options.
    
    return_schedule - bool
    If True, return the schedule computed by the heuristic.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by the heuristic.
    
    If return_schedule == True:
    pi - dict
    The schedule in the form {task : ID of Worker it is scheduled on}.    
    """ 
    
    if return_schedule:
        pi = {}
    
    # List all tasks by upward rank unless alternative is specified.
    if priority_list is None:
        priority_list = dag.critical_path_priorities(direction="upward", cp_type=cp_type, avg_type=avg_type)   
    
    # Schedule the tasks.
    for t in priority_list:         
        # Compute the finish time on all workers and identify the fastest (with ties broken consistently by np.argmin).   
        worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
        min_val = min(worker_finish_times, key=lambda w:w[0]) 
        min_worker = worker_finish_times.index(min_val)                       
        
        # Schedule the task on the chosen worker. 
        ft, idx = min_val
        platform.workers[min_worker].schedule_task(t, finish_time=ft, load_idx=idx)        
        if return_schedule:
            pi[t] = min_worker
                    
    # If schedule_dest, print the schedule to file.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_info(print_schedule=True, filepath=schedule_dest)    
        
    # Compute makespan.
    mkspan = dag.makespan() 
    
    # Save schedule illustration if specified.
    # TODO: some of this stuff needs to be tweaked depending on the DAG/platform, tidy up.
    if schedule_img_dest is not None:
        loads = {}
        for w in platform.workers:
            loads[w.ID] = []
            for t in w.load:
                loads[w.ID].append((t[1], t[2] - t[1]))
        
        fig, ax = plt.subplots(dpi=400)
        for i, w in enumerate(platform.workers):
            ax.broken_barh(loads[w.ID], (5*i, 5), facecolors='#FBC15E', edgecolor='k')
                
        ax.set_xlim(0, mkspan)
        ax.set_xlabel('TIME')
        ax.set_ylabel('PROCESSOR')
        
        ax.set_ylim(0, platform.n_workers * 5)
        ax.set_yticks(list(2.5 + 5*i for i in range(platform.n_workers)))
        ax.set_yticklabels(list("P{}".format(int(w.ID[-1]) + 1) for w in platform.workers)) # +1 for consistency with examples. 
        
        # Add task IDs.
        for i, w in enumerate(platform.workers):
            for t in w.load:
                ax.annotate(t[0], (t[1] + 0.4 * (t[2] - t[1]), 2.5 + 5*i), color='#348ABD')
        
        # Add annotation identifying makespan.
        yf = 0
        for i, w in enumerate(platform.workers):
            if w.load[-1][2] == mkspan:
                yf = i
                break
        ax.annotate('Makespan = {}'.format(mkspan), (mkspan, 5 * (yf + 1)),
            xytext=(0.85, 0.9), textcoords='axes fraction',
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=16,
            horizontalalignment='right', verticalalignment='top')
        
        # Tend to play around with these...
        # ax.set_facecolor('white')  
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        
        plt.savefig('{}/heft_schedule_{}_{}_{}.png'.format(schedule_img_dest, dag.name, platform.name, img_name), bbox_inches='tight') 
        plt.close(fig) 
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset() 
    
    if return_schedule:
        return mkspan, pi    
    return mkspan 

def PEFT(dag, platform, return_schedule=False, schedule_dest=None):
    """
    Predict Earliest Finish Time.
    'List scheduling algorithm for heterogeneous systems by an optimistic cost table',
    Arabnejad and Barbosa, 2014.
    
    Parameters
    ------------------------    
    dag - DAG object.
    Represents the task DAG to be scheduled.
          
    platform - Platform object.
    Represents the target platform. 
    
    priority_list - None/list
    If not None, an ordered list which gives the order in which tasks are to be scheduled. 
        
    return_schedule - bool
    If True, return the schedule computed by the heuristic.
             
    schedule_dest - None/string
    Path to save schedule. 
    
    Returns
    ------------------------
    mkspan - float
    The makespan of the schedule produced by the heuristic.
    
    If return_schedule == True:
    pi - defaultdict(int)
    The schedule in the form {Task : ID of Worker it is scheduled on}.    
    """ 
    
    if return_schedule or schedule_dest is not None:
        pi = {}
    OCT = dag.optimistic_cost_table()
    task_ranks = {t : sum(OCT[t.ID][w.ID] for w in platform.workers) / platform.n_workers for t in dag.top_sort} 
    
    ready_tasks = list(t for t in dag.top_sort if t.entry)    
    while len(ready_tasks):   
        # Find ready task with highest priority (ties broken randomly according to max function).
        t = max(ready_tasks, key=task_ranks.get) 
        # Add optimistic critical path length to finish times and compare.
        worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
        worker_makespans = list(f[0] + OCT[t.ID]["P{}".format(i)] for i, f in enumerate(worker_finish_times)) 
        opt_worker_val = min(worker_makespans)
        opt_worker = worker_makespans.index(opt_worker_val)
        ft, idx = worker_finish_times[opt_worker]
        # Schedule the task.
        platform.workers[opt_worker].schedule_task(t, finish_time=ft, load_idx=idx)          
        if return_schedule or schedule_dest is not None:
            pi[t] = opt_worker 
        # Update ready tasks.                          
        ready_tasks.remove(t)
        for c in dag.graph.successors(t):
            if dag.ready_to_schedule(c):
                ready_tasks.append(c) 
        
    # If schedule_dest, print the schedule to file.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_info(print_schedule=True, filepath=schedule_dest)

    # Compute makespan.
    mkspan = dag.makespan()        
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()  
      
    if return_schedule:
        return mkspan, pi    
    return mkspan 

def HSM(dag, platform, cp_type="WM", rank_avg="WM", CCP=None, priority_list=None, return_schedule=False, schedule_dest=None):
    """
    Used for modified PEFT heuristics proposed in the write up. 
    Could be incorporated into PEFT but decided to use separate function for various reasons.    
    """ 
    
    if return_schedule:
        pi = {}
    
    if CCP is None: 
        CCP = dag.conditional_critical_paths(cp_type=cp_type, lookahead=True)
    
    if priority_list is None:
        priority_list = dag.conditional_critical_path_priorities(platform, CCP=CCP, cp_type=cp_type, rank_avg=rank_avg)
         
    # Schedule the tasks.
    for t in priority_list:         
        # Compute the finish time on all workers and identify the fastest (with ties broken consistently by np.argmin).   
        worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
        worker_makespans = list(f[0] + CCP[t.ID]["P{}".format(i)] for i, f in enumerate(worker_finish_times)) 
        opt_worker_val = min(worker_makespans)
        opt_worker = worker_makespans.index(opt_worker_val)
        ft, idx = worker_finish_times[opt_worker]
        # Schedule the task.
        platform.workers[opt_worker].schedule_task(t, finish_time=ft, load_idx=idx)          
        if return_schedule or schedule_dest is not None:
            pi[t] = opt_worker      
        
    # If schedule_dest, print the schedule to file.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in priority_list:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_info(print_schedule=True, filepath=schedule_dest)

    # Compute makespan.
    mkspan = dag.makespan()        
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()  
      
    if return_schedule:
        return mkspan, pi    
    return mkspan

def CPOP(dag, platform, return_schedule=False, schedule_dest=None):
    """
    Critical-Path-on-a-Processor (CPOP).
    'Performance-effective and low-complexity task scheduling for heterogeneous computing',
    Topcuoglu, Hariri and Wu, 2002.
    """  
    
    if return_schedule or schedule_dest is not None:
        pi = {} 
    
    # Compute upward and downward ranks of all tasks to find priorities.
    _, upward_ranks = dag.critical_path_priorities(direction="upward", return_ranks=True)
    _, downward_ranks = dag.critical_path_priorities(direction="downward", return_ranks=True)
    task_ranks = {t.ID : upward_ranks[t] + downward_ranks[t] for t in dag.graph}     
    
    # Identify the tasks on the critical path.
    ready_tasks = list(t for t in dag.graph if t.entry)  
    cp_tasks = []
    for t in ready_tasks:
        if any(abs(task_ranks[s.ID] - task_ranks[t.ID]) < 1e-6 for s in dag.graph.successors(t)):
            cp = t
            cp_prio = task_ranks[t.ID] 
            cp_tasks.append(cp)
            break        
    while not cp.exit:
        cp = np.random.choice(list(s for s in dag.graph.successors(cp) if abs(task_ranks[s.ID] - cp_prio) < 1e-6))
        cp_tasks.append(cp)
    # Find the fastest worker for the CP tasks.
    worker_cp_times = {w : sum(c.comp_costs[w.ID] for c in cp_tasks) for w in platform.workers}
    cp_worker = min(worker_cp_times, key=worker_cp_times.get)   
       
    while len(ready_tasks):
        t = max(ready_tasks, key=lambda t : task_ranks[t.ID])
        
        if t in cp_tasks:
            cp_worker.schedule_task(t, dag=dag, platform=platform)
            if return_schedule or schedule_dest is not None:
                pi[t] = cp_worker.ID
        else:
            worker_finish_times = list(w.earliest_finish_time(t, dag, platform) for w in platform.workers)
            min_val = min(worker_finish_times, key=lambda w:w[0]) 
            min_worker = worker_finish_times.index(min_val) 
            ft, idx = min_val
            platform.workers[min_worker].schedule_task(t, finish_time=ft, load_idx=idx)        
            if return_schedule or schedule_dest is not None:
                pi[t] = min_worker
    
        # Update ready tasks.                          
        ready_tasks.remove(t)
        for c in dag.graph.successors(t):
            if dag.ready_to_schedule(c):
                ready_tasks.append(c)       
    
    # If schedule_dest, save the priority list and schedule.
    if schedule_dest is not None: 
        print("The tasks were scheduled in the following order:", file=schedule_dest)
        for t in pi:
            print(t.ID, file=schedule_dest)
        print("\n", file=schedule_dest)
        platform.print_info(print_schedule=True, filepath=schedule_dest)       
    
    # Compute makespan.
    mkspan = dag.makespan()        
    
    # Reset DAG and platform.
    dag.reset()
    platform.reset()  
      
    if return_schedule:
        return mkspan, pi    
    return mkspan 