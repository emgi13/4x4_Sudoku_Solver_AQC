#! /usr/bin/env python3
# Imports

import numpy as np
from findiff import FinDiff
from scipy.sparse.linalg import inv, eigs
from scipy.sparse import coo_array, eye, diags, kron
import matplotlib.pyplot as plt
import sys

def expand(val: int, known):
    j = 0
    x = 0
    for i, it in known:
        while (j<i):
            x   <<= 1
            x   +=  val&1
            val >>  1
            j+=1
        x   <<= 1
        x   +=  it
    while(val>0):
        x   <<= 1
        x   +=  val&1
        val >>= 1
    return x

def exchange(n, **kwargs):
    return eye(n, **kwargs)[::-1]

constraints = []
# Row Constraints
for i in [0,1,8,9,16,24]:
    constraints.append((i, i+2, i+4, i+6))
# Column Constraints
for i in range(8):
    constraints.append((i, i+8, i+16, i+24))
# Block Constraints
for i in [0,1,4,5,16,17,20,21]:
    constraints.append((i, i+2, i+8, i+10))

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
        exp_query += f"{bin(val-1)[2:]:0>2}"

# Generate index map for conversion later
ind = -1
index = {}
known_vals = []
for i, it in enumerate(exp_query):
    if it == 'X':
        ind +=1
        index[ind] = i
    else:
        known_vals.append((i, int(it)))
     
# Make Hamiltonians
# Initial Hamiltonian
g   =   10. # strength
H0  =   g*exchange(1<<complexity, dtype = 'D', format='csr')
# Constructing the final hamiltonian
