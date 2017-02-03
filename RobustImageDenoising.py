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

def hamming_dist(x,y):
    """hamming distance between arrays x and y"""
    return abs(x-y).sum()
def denoise(image,error,beta):
    """Denoises image with error rate 'error' and neighbor affinity 'beta' """
    m = int(scipy.sqrt(len(image)))
    
    try: # assert that length of image is a square number
        assert m**2 == len(image)
    except AssertionError:
        raise AssertionError('Length of image should be a square number')
    
    l0 = scipy.log(error/(1-error))#lambda(y=0) = ln(p(0|1)/p(1|0)) = ln(er/(1-er))
    l1 = scipy.log((1-error)/error)
    
    g = nx.Graph()
    
    for i in range(m):
        for j in range(m-1):
            g.add_edge(m*j+i,m*(j+1)+i,cap = beta)
    for i in range(m-1):
        for j in range(m):
            g.add_edge(m*j+i,m*j+i+1,cap = beta)
    g.add_nodes_from(['s','t'])        
    g = g.to_directed()
    
    # 's' is on black nodes side; t is on white nodes side
    for n in g.nodes():
        if n != 's' and n!= 't':
            if image[n] >=0.1: #image[n] ==1 i.e. black side
                g.add_edge('s',n,cap = l1 )
            else:
                g.add_edge(n,'t',cap = -l0 )
    
    value, partition = nx.minimum_cut(g,'s','t',capacity = 'cap')
    
    if 's' in partition[0]:
        black_nodes,white_nodes = partition
    else:
        white_nodes,black_nodes = partition
        
    y = scipy.zeros_like(image,dtype=int) #denoised image
    for n in black_nodes:
        if n != 's':            
            y[n] = 1
    return y

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
    logZ = 0#get_logZ(theta)
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
l0 = er/(1-er)#lambda(y=0) = ln(p(0|1)/p(1|0)) = ln(er/(1-er))
l1 = (1-er)/er
act_image = train_set[160]
act_image[act_image!=0]=1

r = scipy.random.binomial(1,er,size = 784 )
image = act_image + r* (1-2*act_image)

#plt.imshow(image, cmap=plt.cm.gray_r)
#plt.spy(image)
beta = .8
ebeta = scipy.exp(beta) #.....WHY NOT beta/2 ?????

g = nx.Graph()
m = 28
for i in range(m):
    for j in range(m-1):
        g.add_edge(m*j+i,m*(j+1)+i,cap = beta)
for i in range(m-1):
    for j in range(m):
        g.add_edge(m*j+i,m*j+i+1,cap = beta)
facList=[]        
for e1,e2 in g.edges():
    facList.append(PGM.factor([e1,e2],[2,2],[ebeta,1,1,ebeta]))
for n in g.nodes():
    for idx,f in enumerate(facList):
        if n in f.var:
            facList[idx] *= PGM.factor([n],[2],[1,l0 if image[n]<=.1 else l1])
            break
    
F = PGM.FactorList(facList)
theta = PGM.factors2ExpFam(F)
model = MAP_inference_IP(F)
clean_image = scipy.array([int(model.Y[s,s].value) for s in range(m**2)])

nominal = theta.copy()
interval = .1*scipy.absolute(nominal)
model2,lb = solve_IU_robust_MAP(nominal,interval)
robust_image = scipy.array([int(model2.Y[s,s].value) for s in range(m**2)])

clean_image2 = denoise(image,er,beta)
fig,[[ax1,ax2],[ax3,ax4]]= plt.subplots(2,2)
ax1.spy(act_image.reshape(28,28))
ax2.spy(image.reshape((28,28)))
ax3.spy(clean_image.reshape(28,28))
ax4.spy(robust_image.reshape(28,28))
print hamming_dist(clean_image,act_image)
print hamming_dist(robust_image,act_image)

#A_rob = scipy.array([int(model2.Y[s,s].value) for s in range(m**2)])
