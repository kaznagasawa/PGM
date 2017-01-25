# Mesage passing algorithms
# DONE (I think): regular BP,TRW BP
#Next goal: Updating edge appearance probs

import PGM
import scipy
import networkx as nx
#import matplotlib.pyplot as plt
import time
from operator import mul
import cProfile

def is_converged(g):
    """checks if messages have converged, or not"""
    flag = True
    print max(abs(g.edge[s][t]['msg'].val[0] - g.edge[s][t]['old_msg'][0]) for s,t in g.edges() )
    for s,t in g.edges():
        if not scipy.allclose( g.edge[s][t]['msg'].val, g.edge[s][t]['old_msg'],atol=1e-4 ):
            flag = False
            break
    return flag
def is_TRW_converged(g):
    """checks if messages have converged, or not"""
    flag = True
#    print max(abs(g.edge[s][t]['msg'].val[0] - g.edge[s][t]['old_msg'][0]) for s,t in g.edges() )
    for s,t in g.edges():
        if not scipy.allclose( g.edge[s][t]['msg'].val, g.edge[s][t]['old_msg'],atol=1e-4 ):
            flag = False
            break
    return flag
    
def estimate_Z_BP(F,g):
    """estimates the value of Z after message passing has converged
    g is the "directed" graph which contains converged messages"""
    
    # sum(H_s)
    logZ = sum( -sum(g.node[s]['marg']*scipy.log(g.node[s]['marg'])) for s in g.nodes() )
    for s,t in g.edges():
        if s<t:
            val = g.node[s]['marg']/g.edge[t][s]['msg'].val #product over N(s)\t
            msg1 = PGM.factor([s],[F.cardVec[s]],val)

            val = g.node[t]['marg']/g.edge[s][t]['msg'].val
            msg2 = PGM.factor([t],[F.cardVec[t]],val)
            
            marg = msg1*msg2* g.edge[s][t]['factor']        
            marg.val = marg.val/sum(marg.val)
            marg2 = marg.Marginalize(s)*marg.Marginalize(t)
            logZ -= sum(marg.val*scipy.log(marg.val/marg2.val)) #-I_st
            logZ += sum(scipy.log(g.edge[s][t]['factor'].val)*marg.val) #<theta,tau>
                    
    return logZ

def run_BP(F,max_iters = 50):
    g2 = F.g.to_directed()
    for n1,n2 in g2.edges():
        card = F.cardVec[n2]
        g2.edge[n1][n2]['msg'] = PGM.factor([n2],[card],scipy.ones(card))

    for f in F.factors:
        g2.edge[f.var[0]][f.var[1]]['factor'] = f.copy()
        g2.edge[f.var[1]][f.var[0]]['factor'] = f.copy()
    for iters in range(max_iters):
        for s,t in g2.edges(): # to check convergence
            g2.edge[s][t]['old_msg'] = g2.edge[s][t]['msg'].val.copy()
        
        for s in g2.nodes():
            val_s = reduce(mul,[g2.edge[n][s]['msg'].val for n in g2.predecessors(s)])
            for t in g2.predecessors(s):
                val = val_s/g2.edge[t][s]['msg'].val  # product over N(s)\t
                
                #multiply by \phi(s,t)
                msg = PGM.factor([s],[F.cardVec[s]],val) * g2.edge[s][t]['factor']
                msg = msg.Marginalize( scipy.setdiff1d(msg.var,t) )
                msg.val = msg.val/sum(msg.val)
                g2.edge[s][t]['msg'] = msg.copy()
        
        if is_converged(g2):
            print iters
            break
    
    M = []    
    for s in g2.nodes():
        val = reduce(mul, [g2.edge[n][s]['msg'].val for n in g2.predecessors(s)])
        marg = PGM.factor([s],[F.cardVec[s]],val)
        marg.val = marg.val/sum(marg.val)
        g2.node[s]['marg'] = marg.val
        M.append(marg)
    logZ = estimate_Z_BP(F,g2)
    return M,logZ
    
def estimate_Z_TRW(F,g):
    """estimates the value of Z after message passing has converged
    g is the "directed" graph which contains converged messages"""

    g_MST = g.to_undirected() # for finding gradient through MST
    logZ = sum( -sum(g.node[s]['marg']*scipy.log(g.node[s]['marg'])) for s in g.nodes() )
    for s,t in g.edges():
        if s<t:
            val = g.node[s]['marg']/g.edge[t][s]['msg'].val
            msg1 = PGM.factor([s],[F.cardVec[s]],val)

            val = g.node[t]['marg']/g.edge[s][t]['msg'].val
            msg2 = PGM.factor([t],[F.cardVec[t]],val)
            
            marg = msg1 * msg2 * g.edge[s][t]['factor']
            marg.val = marg.val/sum(marg.val)
            
            marg2 = marg.Marginalize(s)*marg.Marginalize(t)
            I_st = sum(marg.val*scipy.log(marg.val/marg2.val))
            logZ -= g.edge[s][t]['p'] * I_st #logZ=<theta,tau> + sum(H_s)- sum(p_st*I_st)
            g_MST.edge[s][t]['grad'] = -I_st
            
            f = g.edge[s][t]['factor']**g.edge[s][t]['p']
            logZ += sum(scipy.log(f.val)*marg.val)
            
    T = nx.minimum_spanning_tree(g_MST,weight='grad')
    return logZ,T
    
def TRW_BP(F,max_iters = 50,damp = .9, g_msg = None):
    """Tree reweighted message passing"""
    g2 = F.g.to_directed()
    for n1,n2 in g2.edges():
        card = F.cardVec[n2]
        g2.edge[n1][n2]['msg'] = PGM.factor([n2],[card],scipy.ones(card)/card)

    for f in F.factors:
        s,t = f.var[0],f.var[1]
        g2.edge[s][t]['factor'] = f**(1.0/g2.edge[s][t]['p'])
        g2.edge[t][s]['factor'] = f**(1.0/g2.edge[t][s]['p'])

    for iters in range(max_iters):
        for s,t in g2.edges():
            g2.edge[s][t]['old_msg'] = g2.edge[s][t]['msg'].val.copy()
        for s in g2.nodes():
            val_s = reduce(mul,[g2.edge[n][s]['msg'].val**g2.edge[n][s]['p'] for n in g2.predecessors(s)])
            for t in g2.predecessors(s):
                val = val_s/g2.edge[t][s]['msg'].val
                msg = PGM.factor([s],[F.cardVec[s]],val)*g2.edge[s][t]['factor']
    
                msg = msg.Marginalize( scipy.setdiff1d(msg.var,t) )
                msg.val = (g2.edge[s][t]['old_msg']**(1-damp) ) * ( msg.val**damp )
                msg.val = msg.val/sum(msg.val)
                g2.edge[s][t]['msg'] = msg.copy()
        
        if is_TRW_converged(g2):
            print iters
            break
    M = []
    for s in g2.nodes():
        val = reduce(mul, [g2.edge[n][s]['msg'].val**g2.edge[n][s]['p'] for n in g2.predecessors(s)])
        marg = PGM.factor([s],[F.cardVec[s]],val)
        marg.val = marg.val/sum(marg.val)
        g2.node[s]['marg'] = marg.val
        M.append(marg)
    logZ,T = estimate_Z_TRW(F,g2)
    return M, logZ,T

def TRW_with_p_update(F,stepsize=.5,iters=10):
    for k in range(iters):
        alpha = stepsize/(k+1)**.7
        M,logZ,T = TRW_BP(F)
        print logZ
        for s,t in F.g.edges():
            F.g.edge[s][t]['p'] = (1-alpha)*F.g.edge[s][t]['p'] + alpha*int((s,t)in T.edges())
    return M,logZ

if __name__ =='__main__':
    pr = cProfile.Profile()
    m = 20
    p = .3
    g = nx.erdos_renyi_graph(m,p,seed=7774) # g should be connected
    while not nx.is_connected(g):
        print 1
        nx.erdos_renyi_graph(m,p)
    fac_list = []
#    scipy.random.seed(1235)
    for e1,e2 in g.edges():
        f = PGM.factor(var=[e1,e2],card=[2,2],val=.1+scipy.rand(4))
        fac_list.append(f)
    F = PGM.FactorList(fac_list)
    
    for s,t in F.g.edges():
        F.g.edge[s][t]['p'] = (g.number_of_nodes()-1.0)/g.number_of_edges()
    
    st = time.time()
    pr.enable()
#    M_BP,logZ_BP = run_BP(F)
    pr.disable()
#    pr.print_stats(sort = 'tottime')
    M_BP,logZ_BP = TRW_with_p_update(F)
    t1 = time.time()-st
    M = F.run_inference(isMax=0)
    M_Z = F.run_inference(isMax=0,findZ=1)
    logZ_act = scipy.log(sum(M_Z[0].val) )
    for s in F.g.nodes():
        print M_BP[s].val[0],M[s].val[0]
#    
    print logZ_BP,logZ_act
#    print t1