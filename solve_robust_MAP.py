# THis file contains code for solving MAP as IP;
# and for solving Robust MAP problem for different types of uncertainty sets

# AIMS: 1) Create IP for solving MAP; DONE!!
#       2) Iterative algorithm for IU Sets: DONE; Aesthetics to be improved
#       3) EU and GBU sets

# Shorthands: IU: interval uncertainty; EU: Ellipsoidal Uncertainty; GBU: Gamma bounded

# NOTE: theta[i,i] for diagonals, but theta[i,j] + theta[j,i] if i!=j

#----CAUTION: converting to exponential family rescales, so it changes Z
#            : model.node_set may not be sorted


"                              ____    _      \
                              ((__))==| |      \
      O              __O__            | |       \
    </\__               \             | |        \
    (    :               )            | |         \
    |\   :              /\            | |          \
    ! !  O             !  !           |_|           \
"

"           ____    _      \
         O ((__))==| |      \
     O  //         | |       \
    | |=           | |        \
    |_|            | |         \
    | \            | |          \
    |  |           |_|           \
"

"        _               \
        ( )               \
        _|_                 \
      /|   |\             \
  ___/_|___|_\__       \
 |\             \       \
 | \_____________\        \
 | |   !   !   | |     \
   |             |     \
   |             |      \
"

"                   _____________________    \
         _         |                     |   \
        ( )        |                     |    \
        _|_        |   HOW TO PASS TIME  |           \
      /| . |\      |   ON A LONG FLIGHT  |         \
     / |_._| \     |_____________________|            \
       |   |            \
       !   !         \
       '   '      \
"


import scipy
import PGM
import pyomo
import pyomo.environ as pe
import networkx as nx

def setup_model(g):
    """sets up the model for solving robust MAP problem"""
        
    model = pe.ConcreteModel()
    
    model.node_set = pe.Set(initialize = [(s,s) for s in g.nodes()])
    model.edge_set = pe.Set(initialize = g.edges(),dimen = 2)
    model.full_set = model.node_set | model.edge_set # union
    
    model.Y = pe.Var(model.full_set,domain = pe.Binary) #Y[i,i] == X[i]
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
    F = PGM.expfam2Factors(theta)
    M = F.run_inference(isMax=0,findZ = 1)
    return scipy.log( sum( M[0].val ) )
    
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
                    for i,j in model.full_set)-logZ >= model.t
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
    while not scipy.isclose(lb,ub): # change tolerance
        model.thetaCons.add(sum( (theta[i,j] + (0 if i==j else theta[j,i]) )*model.Y[i,j]\
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

def get_mu(theta):
    """theta --> mu """
    F = PGM.expfam2Factors(theta)
    M = F.run_inference(isMax=0)
    M2 = F.find_clique_marginal( F.g.edges() )    
    mu = scipy.zeros_like(theta)
    for f in M:
        s = f.var[0].astype(int)
        mu[s,s] = M[s].val[-1]
    for f in M2:
        s,t = f.var[0].astype(int),f.var[1].astype(int)
        if s>t: s,t = t,s
        mu[s,t] = f.val[-1] # last value corresponds to all 1's; therefore = mu
    return mu

def max_A_GBU(model,interval,mu, gamma):
    """finds direction to move by taking first gamma maximum theta's"""
    def get_weight(s,t):
        return scipy.exp(interval[s,t])*abs(mu[s,t]-model.Y[s,t].value)
    w = {(s,t): get_weight(s,t) for s,t in model.full_set }

    max_indices = sorted(w.keys(),key = lambda k: w[k],reverse = True)[:gamma]
    
    direc = scipy.zeros_like(interval)
    for s,t in max_indices:
        direc[s,t] = 1 - 2* model.Y[s,t].value
    return direc
      
def solve_GBU_robust_MAP(nominal,interval,gamma):
    """Solve gamma bounded Robust MAP Problem;nominal: n x n matrix of thetas;
       interval: n x n matrix of intervals around thetas, must be 0 if corresp-
       onding theta is 0; gamma: number of thetas allowed to deviate"""
    F = PGM.expfam2Factors(nominal) # for setting up model
    model = setup_model(F.g)

    ub = scipy.inf
    lb = -scipy.inf
    
    theta = nominal
    logZ = get_logZ(theta)
    mu = get_mu(nominal)
    while not scipy.isclose(lb,ub) and ub > lb: # change tolerance
        model.thetaCons.add(sum( (theta[i,j] + (0 if i==j else theta[j,i]) )*model.Y[i,j]\
                        for i,j in model.full_set) - logZ >= model.t)
        solver = pyomo.opt.SolverFactory('cplex')
        solver.solve(model, tee=False, keepfiles=False,warmstart=True)
        ub = model.OBJ()
        
        direc = max_A_GBU(model,interval,mu,gamma)
        theta = nominal + direc*interval
        logZ = get_logZ(theta)
        lb = max(lb, sum( ( theta[i,j] + (0 if i==j else theta[j,i]) )*model.Y[i,j].value\
                        for i,j in model.full_set) - logZ )               
        print lb,ub
    return model,lb

def solve_EU_robust_MAP(nominal,interval):
    """Solve Ellipsoidal Robust MAP problem;nominal: n x n matrix of  thetas
       P: matrix such that {nominal+P*u|norm(u)<=1} is the uncertainty set
       NOTE: currently testing only for diagonal matrices 'P' """
    F = PGM.expfam2Factors(nominal) # for setting up model
    model = setup_model(F.g)

    ub = scipy.inf
    lb = -scipy.inf
    
    theta = nominal
    logZ = get_logZ(theta)
    mu = get_mu(nominal)
    while not scipy.isclose(lb,ub) and ub > lb: # change tolerance
        model.thetaCons.add(sum(( theta[i,j] + (0 if i==j else theta[j,i]) )*model.Y[i,j]\
                        for i,j in model.full_set) - logZ >= model.t)
               
        solver = pyomo.opt.SolverFactory('cplex')
        solver.solve(model, tee=False, keepfiles=False,warmstart=True)
        ub = model.OBJ()
        
        direc = mu
        for s,t in model.full_set:
            if s>t: s,t = t,s
            direc[s,t] -= model.Y[s,t].value # X[s]*X[t]=1 => direc[s,t] = -1
        dummy = interval*direc
        norm = scipy.linalg.norm(dummy[dummy!=0])
        theta = nominal + (interval**2)*direc/norm
        
        logZ = get_logZ(theta)
        lb = max(lb,sum( ( theta[i,j] + (0 if i==j else theta[j,i]) )*model.Y[i,j].value\
                        for i,j in model.full_set) - logZ )
        print lb,ub

if __name__ == '__main__':
    
    n=20
    g = nx.erdos_renyi_graph(n,.3,seed=7774) # g should be connected
    fac_list = []
    scipy.random.seed(1235)
    for e1,e2 in g.edges():
        f = PGM.factor(var=[e1,e2],card=[2,2],val=scipy.rand(4))
        fac_list.append(f)
    F = PGM.FactorList(fac_list)
    theta = PGM.factors2ExpFam(F)
    A,max_p = PGM.greedy_MAP_assignment(theta,random_runs = 20,heur = 'best')
#    F2 = PGM.expfam2Factors(theta)
    case = 1
    if case == 1:
        model = MAP_inference_IP(F,warmstart = True)
        A_act = scipy.array([int(model.Y[s,s].value) for s in range(n)])
        print model.OBJ()
#    if case == 2:
        nominal = theta.copy()
        interval = .1*scipy.absolute(nominal)
        model,lb = solve_IU_robust_MAP(nominal,interval)
        A_IU = scipy.array([int(model.Y[s,s].value) for s in range(n)])
        print sum(abs(A_IU-A_act))
    if case == 3:
        nominal = theta.copy()
        interval = .1*scipy.absolute(nominal)
        solve_EU_robust_MAP(nominal,interval)
    if case == 4:
        nominal = theta.copy()
        interval = .1*scipy.absolute(nominal)
        obj=[]
        for k in range(2,10):
            gamma = k
            model,lb = solve_GBU_robust_MAP(nominal,interval,gamma)
            print scipy.array([int(model.Y[s,s].value) for s in range(n)])