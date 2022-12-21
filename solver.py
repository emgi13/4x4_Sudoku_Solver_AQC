#! /usr/bin/env python3
# Imports

import numpy as np
from findiff import FinDiff
from scipy.sparse.linalg import inv, eigs
from scipy.sparse import csr_matrix, eye, diags, kron
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

# Generate X operator for the ith bit, n dim
def x_op(i: int, n: int):
    ival = 1<<(i)
    row = [(high_bits<<(i+1))+ival+low_bits for low_bits in range(1<<(i)) for high_bits in range(1<<(n-i-1))]
    col = [(high_bits<<(i+1))+ival+low_bits for low_bits in range(1<<(i)) for high_bits in range(1<<(n-i-1))]
    data = [1 for _ in range(1<<(n-1))]
    return csr_matrix((data, (row, col)), dtype='D')
    
constraints = []
# Row Constraints
for i in [0,1,8,9,16,17,24,25]:
    constraints.append((i, i+2, i+4, i+6))
# Column Constraints
for i in range(8):
    constraints.append((i, i+8, i+16, i+24))
# Block Constraints
for i in [0,1,4,5,16,17,20,21]:
    constraints.append((i, i+2, i+8, i+10))

def cost(a,b,c,d):
    global  known_vals_dict, complexity
    print(a, b, c, d)
    I = eye(1<<complexity)
    unk = 0
    if a in known_vals_dict.keys():
        A = known_vals_dict[a]*I
    else:
        A = x_op(index[a], complexity)
        unk+=1
    if b in known_vals_dict.keys():
        B = known_vals_dict[b]*I
    else:
        B = x_op(index[b], complexity)
        unk+=1
    if c in known_vals_dict.keys():
        C = known_vals_dict[c]*I
    else:
        C = x_op(index[c], complexity)
        unk+=1
    if d in known_vals_dict.keys():
        D = known_vals_dict[d]*I
    else:
        D = x_op(index[d], complexity)
        unk+=1
    print(unk)
    z = (A+B+C+D-2*I)
    return z*z

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
print(exp_query)
# Generate index map for conversion later
ind = -1
index = {}
known_vals = []
known_vals_dict = {}
for i, it in enumerate(exp_query):
    if it == 'X':
        ind +=1
        index[i] = ind
    else:
        known_vals.append((i, int(it)))
        known_vals_dict[i] = int(it)
# Make Hamiltonians
# Initial Hamiltonian
g   =   10. # strength
H0  =   g*exchange(1<<complexity, dtype = 'D', format='csr')
# Constructing the final hamiltonian
Hf = cost(*constraints[0])
for a,b,c,d in constraints[1:]:
    Hf += cost(a,b,c,d)
    
