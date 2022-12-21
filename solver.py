#! /usr/bin/env python3
# Imports

import numpy as np
from findiff import FinDiff
from scipy.sparse.linalg import inv, eigs
from scipy.sparse import coo_array, eye, diags, kron
import matplotlib.pyplot as plt





def main():
    # Get input
    query = ''
    while(True):
        query += input().strip()
        if len(query)>=16: break
    if len(query) != 16: raise Exception("Input must have 16 charachters")
    print(query)
    
    # Expand Query
    exp_query = ''
    complexity = 0
    for it in query:
        if it == 'X':
            exp_query+='XX'
            complexity +=2
        else:
            try:
                val = int(it)
            except:
                raise Exception(f"Unknown Input {it}. Must be X, 1, 2, 3 or 4.")
            if val>4 or val<1: raise Exception(f"Unknown Input {it}. Must be X, 1, 2, 3 or 4.")
            exp_query += bin(val-1)[2:]
    print(exp_query, complexity)
    
    # Generate index map for conversion later
    ind = -1
    index = {}
    for i, it in enumerate(exp_query):
        if it == 'X':
            ind +=1
            index[ind] = i
    print(index)
    
     


if __name__ == '__main__':
    main()