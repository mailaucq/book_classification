#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:20:13 2020

@author: usuario
"""
import json
from igraph import *;




def igraph2json1(g, fileName='test.json'):
    N = g.vcount()
    E = g.ecount()
    edges_dict = {}
    edges_list = []
    for i in range(E):		
        edge = g.es[i].tuple
        edges = [int(g.vs[edge[0]]["name"]), int(g.vs[edge[1]]["name"])]
        edges_list.append(edges)
    edges_dict["edges"] = edges_list
    
    with open(fileName, 'w') as outfile:
        json.dump(edges_dict, outfile)
    