#! /usr/bin/env python3
# Imports

import numpy as np
from findiff import FinDiff
from scipy.sparse.linalg import inv, eigs
from scipy.sparse import csr_matrix, eye, diags, kron
import matplotlib.pyplot as plt
import sys
import gc
from tqdm import tqdm

fno = 0
def plot(state, dt, it, t):
    global fno, fig
    fno+=1
    plt.ylim((-14, 0.5))
    plt.xlabel(r"$x_{10}$")
    plt.ylabel(r"$\log(Prob(x))$")
    plt.grid()
    plt.plot(2*np.log(np.abs(state)))
    plt.title(f"dt={dt:.3e} it={it:8d} t={t:6.2f}")
    plt.savefig(f"Images/{fno:05d}.jpg", dpi=240, bbox_inches = 'tight')
    plt.clf()
    plt.close()
 
def expand(val: int, known):
    j = 0
    x = 0
    for i, it in known:
        while (j<i):
            x = 2*x + val%2
            val = val//2
            j+=1
        x = 2*x + it
        j+=1
    while(val>0):
        x = 2*x + val%2
        val = val//2
    return x

def sum_sx(N:int):
    I = eye(2, format='csr')
    sx = csr_matrix([[0,1],[1,0]], dtype = int)
    Z = sx
    for i in range(1, N):
        Z = csr_matrix(kron(Z, I) + kron(eye(1<<i, format='csr'), sx))
    return Z


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
    global  known_vals_dict, complexity, I
    if a in known_vals_dict.keys():
        A = known_vals_dict[a]*I
    else:
        A = x_op(index[a], complexity)
    if b in known_vals_dict.keys():
        B = known_vals_dict[b]*I
    else:
        B = x_op(index[b], complexity)
    if c in known_vals_dict.keys():
        C = known_vals_dict[c]*I
    else:
        C = x_op(index[c], complexity)
    if d in known_vals_dict.keys():
        D = known_vals_dict[d]*I
    else:
        D = x_op(index[d], complexity)
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

I = eye(1<<complexity)

# Make Hamiltonians
# Initial Hamiltonian
g   =   2. # strength
H0  =   g*sum_sx(complexity)
# Constructing the final hamiltonian
a,b,c,d = constraints[0]
Hf = cost(a,b,c,d)
for a,b,c,d in constraints[1:]:
    Hf += cost(a,b,c,d)
# General time Hamiltonian
def H(s):
    return (1-s)*H0 + s*Hf

def U(s, dt):
    forward = (I-1.j*dt*H(s))
    # backward = (I+0.5j*dt*H(s))
    return forward

# Simulate Evolution by Euler approximation

# Initialize state
pauli_x_ground = 1/np.sqrt(2)*np.matrix([1,-1], dtype='D').H
state = pauli_x_ground
for _ in range(complexity-1):
    state = np.kron(state, pauli_x_ground)
    
T = 100
ts, dt = np.linspace(0, T, 10_000, retstep=True, dtype='d')

for it in tqdm(range(len(ts))):
    state = U(ts[it]/T, dt)*state
    # Renormalize
    state = state/np.sqrt((state.H*state))
    if it%13 == 1: plot(state, dt, it, ts[it])

def reencode(x):
    xb = bin(x)[2:].zfill(32)
    xb = list(xb)
    ren = ''
    while xb:
        a = xb.pop(0)
        b = xb.pop(0)
        val = 2*int(a) + int(b) + 1
        ren += str(val)
    return ren

# Sort Results
sol_sort = []
prob_dist = np.power(np.abs(state), 2)  
avg_prob = np.average(prob_dist)
Solutions = np.where(prob_dist>avg_prob)[0]
for sol in Solutions:
    sol_sort.append((sol, expand(sol, known_vals), reencode(expand(sol, known_vals)),prob_dist[sol][0,0]))

sol_sort.sort(key = lambda x:x[-1], reverse=True)
print(f"{0:6d} : {0:20d} : {bin(0)[2:].zfill(32)} : {query} : {0:.8f}")
print("-"*100)
for a,b,c,d in sol_sort:
    print(f"{a:6d} : {b:20d} : {bin(b)[2:].zfill(32)} : {c} : {d:.8f}")
