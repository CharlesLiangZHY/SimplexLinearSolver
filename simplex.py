import numpy as np

import sys 
import time

'''
Given CSV path
return numpy array
'''
def loadCSV(path):
    data = np.loadtxt(path, delimiter=",")
    return data


'''
standard form linear programming problem
min c'x  s.t. Ax = b, x >= 0
where A: mxn ; c: nx1 ; b: mx1
'''
class stdLP():
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c
        self.m = A.shape[0] # number of constraints 
        self.n = A.shape[1] # number of variables

        if self.m != b.shape[0] or self.n != c.shape[0]:
            print("Error: Wrong Dimension")

    def auxiliaryProblem(self):
        aux_A = np.concatenate((self.A, np.identity(self.m)), axis = 1)
        aux_c = np.concatenate( (np.zeros(self.n), np.ones(self.m)) )
        return stdLP(aux_A, b, aux_c) 

'''
Given standard form of LP
return the simplex table
'''
def simplexTable(LP):
    m = LP.m
    n = LP.n
    table = np.zeros((m + 1, n + 1))
    table[:m, :n] = LP.A
    table[-1, :n] = LP.c
    table[:m, -1] = LP.b
    return table

'''
multiplier * row_i + row_j -> row_j
operation on numpy array is pointer operation (which means it will replace the origin)
'''
def rowOperation(matrix, multiplier, i, j):
    if j != i:
        matrix[j,:] = matrix[j,:] + multiplier*matrix[i,:]
    else:
        matrix[i,:] =  multiplier*matrix[i,:]

### Gaussian elimination
def eliminate(table, p, q): ### pivot is (p,q)
    col_q = table[:, q] # the last row also needs elimination 
    rowOperation(table, 1/col_q[p], p, p)
    for i in range(len(col_q)):
        if i == p:
            continue
        multiplier = -col_q[i]
        rowOperation(table, multiplier, p, i)


def solve(table, debug=False):

    def checkOptimal():
        for r in table[-1,:-1]:
            if r < 0:
                return False
        return True

    def findPivot(q):
        col_q = table[:-1, q]
        p = -1
        entry = float("inf")
        for i in range(len(col_q)):
            if col_q[i] <= 0:
                continue
            if table[i, -1]/col_q[i] < entry: ### ensure the last column always larger than 0
                entry = table[i, -1]/col_q[i]
                p = i
        return p

    while not checkOptimal(): ### do eliminataion
        q = np.argmin(table[-1,:-1])
        p = findPivot(q)
        if p == -1: ### fail to find a pivot in this column
            print(table)
            print("Pivot not found.")
            break
        else:
            eliminate(table, p, q)
        if debug:
            print(table)



def interpret(table): ### interpret the table

    def findBasis(col):
        m = table.shape[0]-1
        n_zeros = 0
        basis = -1
        for i in range(m):
            if table[i, col] == 0:
                n_zeros += 1
            elif table[i, col] == 1:
                basis = i
        if n_zeros == m-1 and basis != -1: ### basis column
            return basis
        else:
            return -1
    
    n = table.shape[1]-1
    bases = np.zeros(n, dtype=np.int_) - 1 ### initialize with -1 and the index must be integer
    solution = np.zeros(n)
    for i in range(n):
        basis = findBasis(i)
        if basis != -1:
            solution[i] = table[basis , -1]
            bases[i] = basis ### the location of 1

    f = -table[-1,-1] # minimum

    return bases, solution, f


'''
if the LP is satisfiable return an initial setting
else return None
'''
def firstPhase(LP):

    def checkBases(bases): ### check whether there is auxiliary variable in the base
        for i in range(LP.n, LP.n+LP.m):
            if bases[i] != -1:
                return i-LP.n ### return the index of the auxiliary variable

        ### all auxiliary variables have exited bases
        return -1

    auxP = LP.auxiliaryProblem()
    table = simplexTable(auxP)

    ### initialize the table, force auxiliary variables to exit bases
    for i in range(A.shape[0]):
        rowOperation(table,-1,i,-1) 
    
    
    solve(table)

    bases, solution, f = interpret(table)

    

    if f > 1e-10: ### for numerical tolerance
        return None ### the origin problem is unsatisfiable 
    else: ### remove the redundant constraints
        redundance = []
        while checkBases(bases) != -1: ### there is still auxiliary varible in bases
            aux = checkBases(bases) # varible index
            p = bases[aux+LP.n] # row i.e constraint index

            alternative_basis = table[p,:LP.n]
            q = -1
            for i in range(len(alternative_basis)):
                if alternative_basis[i] == 0:
                    continue
                else:
                    q = i
                    break
            if q == -1: ### this auxiliary variable cannot be substituted, which implies this contraint is redundant
                redundance.append(p)
            else:

                eliminate(table, p, q)
                

            bases[aux+LP.n] = -1 ### mark that this auxiliary variable is done

    table = np.delete(table, redundance, axis=0) # delete redundant constraints

    table = np.delete(table, [i for i in range(LP.n,LP.m+LP.n)], axis=1) # delete auxiliary variables
 
    table[-1,:-1] = LP.c


    return table


def simplex(LP):

    table = firstPhase(LP)

    if table is None:
        print("The constraints can not be satisfied. No solution.")
        return None, None
    else:
        ### reform to canonical form
        bases, _, _ = interpret(table)
        for i,j in enumerate(bases):
            if j == -1:
                continue
            else:
                rowOperation(table,-table[-1,i], j,-1) 
        
        solve(table)

        _, solutions, f = interpret(table)
        print("The solution is ", solutions)
        print("The optimal is ", f)

        return solutions, f




if __name__ == '__main__':

    path = None

    for i in range(len(sys.argv)):
        if sys.argv[i] == '-p':
            path = sys.argv[i+1]

    if path is not None:

        A = loadCSV(path+"/A.csv")
        b = loadCSV(path+"/b.csv")
        c = loadCSV(path+"/c.csv")
        
        LP = stdLP(A,b,c)
     
        solution, optimal = simplex(LP)

    else:
        print("Usage: python3 simplex -p path_to_datafolder")
        