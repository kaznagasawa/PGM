# Image Denoising with robust MAP 
# Z is calculated using regular Mesage Passing

import PGM
import networkx as nx
import scipy
import MessagePassing as BP
import pyomo.environ as pe
import pyomo

import gzip
import cPickle
import matplotlib.pyplot as plt

def setup_model(g):
    """sets up the model for solving robust MAP problem"""
        
    model = pe.ConcreteModel()
    
    model.node_set = pe.Set(initialize = [(s,s) for s in g.nodes()])
    model.edge_set = pe.Set(initialize = g.edges(),dimen = 2)
    model.full_set = model.node_set | model.edge_set # union
    
    model.Y = pe.Var(model.full_set,domain = pe.Binary) #Y[i,i] == X[i]
#    model.Y = pe.Var(model.full_set,domain = pe.Reals,bounds=(0,1)) #Y[i,i] == X[i]
    model.t = pe.Var(domain = pe.Reals)

    def defY1_rule(model,i,j):
        return model.Y[i,j]<= model.Y[i,i]    
    model.defYcons1 = pe.Constraint(model.edge_set, rule = defY1_rule)
    def defY2_rule(model,i,j):
        return model.Y[i,j]<= model.Y[j,j]    
    model.defYcons2 = pe.Constraint(model.edge_set, rule = defY2_rule)
    def defY3_rule(model,i,j):
        return model.Y[i,j]>= model.Y[i,i] + model.Y[j,j] - 1    
    model.defYcons3 = pe.Constraint(model.edge_set, rule = defY3_rule)

    model.OBJ = pe.Objective(expr = model.t, sense = pe.maximize)
    model.thetaCons = pe.ConstraintList()
    
    return model

def get_logZ(theta):
    M,logZ = BP.run_BP(F)
    return logZ
    
def MAP_inference_IP(F,warmstart = False):  
    """converts F to exponential family form; then finds MAP assignment 
       using IP formulation"""
       
    model = setup_model(F.g)
    
    theta = PGM.factors2ExpFam(F)  
    logZ = get_logZ(theta)
    if warmstart:
        A,max_p = PGM.greedy_MAP_assignment(theta)
        for s,t in model.full_set:
            model.Y[s,t].value = int(A[s]*A[t])
    
    def t_rule(model):# logZ term to compare objective with robust case
        return sum( ( theta[i,j] + (0 if i==j else theta[j,i]) )*model.Y[i,j]\
                    for i,j in model.full_set) -logZ>= model.t
    model.t_cons = pe.Constraint(rule = t_rule)
    
    solver = pyomo.opt.SolverFactory('cplex')
    solver.solve(model, tee=False, keepfiles=False,warmstart=warmstart)
    
    return model
    
def solve_IU_robust_MAP(nominal,interval):
    """Solve IU Robust MAP Problem;nominal: n x n matrix of thetas;
       interval: n x n matrix of intervals around thetas, must be 0 if corresp-
       onding theta is 0"""
    F = PGM.expfam2Factors(nominal) # for setting up model
    model = setup_model(F.g)
    ub = scipy.inf
    lb = -scipy.inf
    
    theta = nominal + interval # direc=all ones
    logZ = get_logZ(theta)
    while not ( ub<lb and scipy.isclose(lb,ub,rtol = 0.01) ): # change tolerance
        model.thetaCons.add( sum( ( theta[i,j] + (0 if i==j else theta[j,i]) )*model.Y[i,j]\
                        for i,j in model.full_set) - logZ >= model.t)
        solver = pyomo.opt.SolverFactory('cplex')
        solver.solve(model, tee=False, keepfiles=False,warmstart=True)
        ub = model.OBJ()

        # phi_i = 0 --> direc_i = 1; phi_i = 1 --> direc_i = -1; therefore direc = 1-2*phi
        direc = scipy.zeros_like(nominal)
        for s,t in model.full_set:
            if s>t: s,t = t,s
            direc[s,t] = 1 - 2*model.Y[s,t].value
        theta = nominal + direc*interval
        logZ = get_logZ(theta)
        lb = max(lb,sum( ( theta[i,j] + (0 if i==j else theta[j,i]) )*model.Y[i,j].value\
                        for i,j in model.full_set) - logZ )
        print lb,ub
    return model,lb

ds = cPickle.load( gzip.open('mnist.pkl.gz') )

train_set,_ = ds[0]

er = .1
act_image = train_set[150]
act_image[act_image!=0]=1
act_image = act_image.reshape([28,28])

r = scipy.random.binomial(1,er,size = (28,28) )
image = act_image + r* (1-2*act_image)

#plt.imshow(image, cmap=plt.cm.gray_r)
#plt.spy(image)
size = 28
beta = .8

g = nx.Graph()
m = 28
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

F = PGM.FactorList(facList)
theta = PGM.factors2ExpFam(F)
model = MAP_inference_IP(F)
A_act = scipy.array([int(model.Y[s,s].value) for s in range(m**2)])

nominal = theta.copy()
interval = .1*scipy.absolute(nominal)
model2,lb = solve_IU_robust_MAP(nominal,interval)
A_rob = scipy.array([int(model2.Y[s,s].value) for s in range(m**2)])
#M,logZ = BP.run_BP(F)

#for n in g.nodes():
#    if n != 's' and n!= 't':
#        if image[n] <=0.1: #image[n] ==0
#            g.add_edge('s',n,cap = scipy.log( (1-er)/er) )
#            facList.append(PGM.factor() )
#        else:
#            g.add_edge(n,'t',cap = scipy.log( (1-er)/er))
#
#value, partition = nx.minimum_cut(g,'s','t',capacity = 'cap')
#
y1 = scipy.zeros(784)
y2 = scipy.zeros(784)
for s in range(784):
    if model.Y[s,s].value>=.9:
        y1[s] = 1
    if model2.Y[s,s].value>=.9:
        y2[s] = 1
y1 = y1.reshape([28,28])
y2 = y2.reshape([28,28])
#
fig,[ax1,ax2]= plt.subplots(1,2)
ax1.spy(y1)
ax2.spy(y2)
#ax3.spy(y)

#print scipy.absolute(y-act_image.reshape([28,28])).sum() / 784.0