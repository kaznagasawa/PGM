import scipy
import networkx as nx
import PGM

g = nx.Graph()
m = 28
beta = 0.8
for i in range(m):
    for j in range(m-1):
        g.add_edge(m*j+i,m*(j+1)+i)
for i in range(m-1):
    for j in range(m):
        g.add_edge(m*j+i,m*j+i+1)
facList=[] 
for e1,e2 in g.edges():
    rnd = scipy.rand()
    g.edge[e1][e2]['cap'] = beta*(1 if rnd >.5 else -1)
    facList.append(PGM.factor([e1,e2],[2,2],[1,1,1,scipy.exp(beta*(1 if rnd >.5 else -1))]))

m = m**2
F = PGM.FactorList(facList)

theta = PGM.factors2ExpFam(F)

#---------------
A = theta/4 # for nondiagonal terms
scipy.fill_diagonal (A, .25* sum(theta+theta.T) )
A = scipy.linalg.block_diag(0,A)
A = -(A + A.T)
d = 1.0/3* scipy.array([3]+[4]*m)

def get_feasible_soln(A):
    """generates x such that A+diag(x) is psd"""
    eig,_ = scipy.linalg.eig(A)
    min_eig = min(eig)
    x = (-min_eig+1)*scipy.ones(A.shape[0]) if min_eig<0 else scipy.zeros(A.shape[0])
    return x
def obj_val(A,x,m,d):
    return -(m+1) - scipy.log(scipy.linalg.det(A+scipy.diag(x))) + d.T.dot(x)

def is_psd(A,x=[0]):
    try:
        scipy.linalg.cholesky(A+scipy.diag(x))
        return True
    except scipy.linalg.LinAlgError:
        return False
def derivs(A,x,d):
    inv_mat = scipy.linalg.inv(A+scipy.diag(x))
    grad = -scipy.diag(inv_mat) + d
    hess = inv_mat**2
    return grad,hess

alpha=.3
beta=0.5
tol = 1e-5

x0 = get_feasible_soln(A)
x = scipy.array(x0).copy()
z0 = obj_val(A,x,m,d)
obj_vals=[z0]
grad,hess = derivs(A,x,d)
direc = - scipy.linalg.solve( hess, grad )
lbd = grad.dot(direc)

#for i in range(10):
while not lbd**2 <= tol:
    
    t = 1 # step size
    while (obj_val(A,x+t*direc,m,d) > z0 + alpha*t*lbd or not is_psd(A,x+t*direc)):
       t=beta*t   # backtracking line search
    x += t*direc
    z0 = obj_val(A,x,m,d)
    obj_vals.append(z0)
    grad,hess = derivs(A,x,d)
    direc = -scipy.linalg.solve( hess, grad )
    lbd = grad.dot(direc)
    print t

ub = obj_val(A,x,m,d)
ub = .5*( ub + m*scipy.log(1.0/4) ) + .5*m*scipy.log(2*scipy.pi*scipy.e) # adjusting entropy terms
ub += ( theta.sum() + theta.trace() )/4 # change of random variable terms

     
#---solving exact problems--- Ising formulation-- X\in{0,1}
#import cvxpy as cvx
#import itertools
#import time
#tau = cvx.Variable(m,m)
#Z = cvx.hstack( cvx.vstack(1,cvx.diag(tau)), cvx.vstack(cvx.diag(tau).T,tau) )
#cons = [ 0<=tau,tau<=1,tau == tau.T ]
#cons.extend([0 <= tau[i,i]+tau[j,j]-tau[i,j]     for i,j in itertools.combinations(range(m),2) ])
#cons.extend([     tau[j,j] >= tau[i,j] for i,j in itertools.combinations(range(m),2) ])
#cons.extend([     tau[i,i] >= tau[i,j] for i,j in itertools.combinations(range(m),2) ])
#
#blkd = 1.0/12*scipy.diag([0]+[1]*m)
#s = sum(theta[i,j]*tau[i,j] for i in range(m) for j in range(m))
#obj =cvx.Maximize( s + .5*cvx.log_det(Z+blkd))
#
#prob = cvx.Problem(obj,cons)
#st = time.time()
#prob.solve()
#t1 = time.time()-st
#
#Z_ub = obj.value + m/2.0*scipy.log(2*scipy.pi*scipy.e)
#
#F2 = PGM.expfam2Factors(theta)
#M = F2.run_inference(isMax=0,findZ=1)
#Z_act = scipy.log( sum(M[0].val) )
#
#print Z_ub,Z_act

# Solving exact----- X \in {-1,1}

#lmda = cvx.Variable(m+1)
#obj = cvx.Minimize( -(m+1) - cvx.log_det(A+cvx.diag(lmda)) + lmda.T*d )
#cons = [A+cvx.diag(lmda)>>0]
##Y = cvx.Variable(m+1,m+1)
##obj = cvx.Maximize(-cvx.trace(A*Y) + cvx.log_det(Y))
##cons = [cvx.diag(Y)==d]
#
#prob = cvx.Problem(obj,cons)
#st = time.time()
#sol = prob.solve()
#t2 = time.time()-st
#sol = .5*( sol + m*scipy.log(1.0/4) ) + .5*m*scipy.log(2*scipy.pi*scipy.e) # adjusting entropy terms
#sol += ( theta.sum() + theta.trace() )/4 # change of random variable terms
#print sol, Z_act