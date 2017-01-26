import PGM
import scipy
import networkx as nx
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

import find_approx_factors_expoFam

def getCondDistn(F,i,A):
    """Gives conditional distribution of node 'i' of cardinality 'card' 
       given assignment 'A' """   
    card = F.cardVec[i]
    #make array for each assignment of 'i' and other elements being from 'A'
    A_array = scipy.repeat([A],card,axis = 0)
    A_array[:,i] = range(card)
    dist = sum(scipy.log(f.val[PGM.A2I(A_array[:,f.var],f.card)])
           for idx,f in enumerate(F.factors) if idx in F.var2facs[i]) # sum of all log factors for relevant assignments
    dist = scipy.exp(dist)
    return dist/sum(dist)
    
def getSample(F,A):
    """perform 1 iteration of Gibbs to get next sample
       A:assignment;card: cardinality vector"""
    card = F.cardVec
    for i in range(len(card)):
        dist = getCondDistn(F,i,A)
        a, = scipy.random.choice( card[i], 1, p = list(dist) )
        A[i] = a
    return A
      
def Gibbs(F, n = 5000,burn = 1000):
    """provides gibbs samples from the graphical model given by factors F
       n = number of Gibbs samples; burn = number of samples burned"""
    A = scipy.zeros_like(F.cardVec)
    samples=[]
    for k in range(burn): # burn these samples
        A = getSample(F,A)        
    for k in range(n):
        if k%100 == 0:
            print k
        A = getSample(F,A)
        samples.append( A.copy() )
    return scipy.array(samples).astype(float)

def estimate_Z(F,n=5000):
    """estimate Z from uniformly distributed samples
       returns array of f(x)*prod(card) for all samples"""
    card = F.cardVec
    p_hat = scipy.zeros(n)
    A = scipy.random.choice(2,[n,len(card)]) # change to incorporate different cardinalities
    p_hat = sum(scipy.log([f.val[PGM.A2I(A[:,f.var],f.card)] for f in F.factors]))#unnormalized prob
    return scipy.exp(p_hat)*card.prod()

def estimate_Z_imp_sampling(F,G,n):
    """estimates Z_F using graph G; First collects 'n' gibbs samples from G;
       returns array of f(x)/g(x) for all samples"""    
    samples = Gibbs(G,n)
    f_over_g_samples = sum(scipy.log([f.val[PGM.A2I(samples[:,f.var],f.card)] for f in F.factors]))\
                     - sum(scipy.log([f.val[PGM.A2I(samples[:,f.var],f.card)] for f in G.factors]))
    return scipy.exp(f_over_g_samples)

def plot_Z_convergence(Z_act,*args,**kwargs):
    """Plots Z convergence; *args takes samples; labels can be provided in kwargs as list named 'label'; 
       if labels are provided, then len(*args) should be = to len(label)"""
    legend = 0
    if kwargs.has_key('label'):# plot the legend iff labels provided
        assert len(args) == len(kwargs['label'])
        legend = 1
        
    else: kwargs['label'] = [' '*len(args)]
    fig,ax = plt.subplots(1,1)
    for samples,label in zip(args,kwargs['label']):
        n = len(samples)
        run_avg = samples.cumsum()/scipy.arange(1,n+1)
        
        ax.plot(scipy.log(run_avg[200:]),label = label)
    ax.axhline(scipy.log(Z_act),color = 'r',label = r'Actual $Z$')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Running average (log scale)')
    ax.set_title(r'Estimation of $Z$ (Partition Function)')
    if legend ==1: ax.legend(loc = 'best')
    return fig,ax
        
def plot_marginal_convergence(samples,M,node=0):
    """plots convergence of the gibbs samples;samples: assignments
       M: exact marginals after inference of F; node: node to be plotted"""
    n = samples.shape[0]
    run_avg = samples[:,node].cumsum(axis=0)/scipy.arange(1,n+1)
    
    fig,ax = plt.subplots(1,1)
    ax.plot(run_avg,label = r'Approximate $P(X_{%d}=1)$ from Gibbs samples' %node)
    ax.axhline(M[node].val[1],color = 'r',label = r'Exact $P(X_{%d}=1)$' %node)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Running average')
    ax.legend(loc = 'best')
    return fig,ax

def plot_marginal_convergence2(samplesF,samplesG,F,G,M,node=0):
    """plots convergence of the gibbs samples using samplesF and importance
       sampling using samplesG; node: node to be plotted"""
    n = samplesF.shape[0]
    run_averageF = samplesF[:,node].cumsum()/scipy.arange(1,n+1)

    f_over_g_samples = sum(scipy.log([f.val[PGM.A2I(samplesG[:,f.var],f.card)] for f in F.factors]))\
                     - sum(scipy.log([f.val[PGM.A2I(samplesG[:,f.var],f.card)] for f in G.factors]))
    f_over_g_samples = scipy.exp(f_over_g_samples)
    samples2 = samplesG[:,node]*f_over_g_samples
    run_averageG = samples2.cumsum()/f_over_g_samples.cumsum()
    
    fig,ax = plt.subplots(1,1)
    ax.plot(run_averageF,label = r'Approximate $P(X_{%d}=1)$ from Gibbs samples' %node)
    ax.plot(run_averageG,label = r'Approximate $P(X_{%d}=1)$ from Importance Sampling' %node)
    prob = M[node].val[1]/sum(M[node].val)
    ax.axhline(prob,color = 'r',label = r'Exact $P(X_{%d}=1)$' %node)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Running average')
    ax.legend(loc = 'best')
    return fig,ax
        
if __name__ == '__main__':   
    g = nx.erdos_renyi_graph(20,.3,seed=7774) # g should be connected
#    scipy.random.seed(38422)
    fac_list = []
    for e1,e2 in g.edges():
        f = PGM.factor(var=[e1,e2],card=[2,2],val=2*scipy.rand(4))
#        f.val[3] = (1+.4*scipy.rand()) * f.val[1] * f.val[2] / f.val[0]
        fac_list.append(f)
    scipy.random.seed()    
    F = PGM.FactorList(fac_list)
    n = 10000
    card = F.cardVec
    
    M2 = F.run_inference(isMax=1)
    M = F.run_inference(isMax = 0,findZ = 1)
    Z_act = sum(M[0].val)
    exact = scipy.array([m.val[1] for m in M])

    tree,model,G = find_approx_factors_expoFam.convert_to_tree(F,'greedy')
    M_G = G.run_inference(isMax = 0,findZ = 1)
    Z_G = sum(M_G[0].val)
    
    samplesF = estimate_Z(F,n)
    samplesG = estimate_Z_imp_sampling(F,G,n)
    samplesG = samplesG*Z_G
#    Z_est_F = scipy.mean(samplesF)
#    Z_est_G = scipy.mean(samplesG)
    fig,ax = plot_Z_convergence(Z_act,samplesG,samplesF,label = ['Importance Sampling','Monte Carlo estimation'])
    fig.savefig('Graphs/estimateZ')
#    fig,ax = plot_Z_convergence(Z_act,samplesG,label = ['Importance Sampling'])
#    fig.savefig('Graphs/estimateZ_imp')
    samplesG = Gibbs(G,n)
    M_G = G.run_inference(isMax = 0,findZ = 0)
    fig,ax = plot_marginal_convergence(samplesG,M_G)
#    fig.savefig('Graphs/gibbs_marginal')
    samplesF = Gibbs(F,n)
    samplesG = Gibbs(G,n)
    plot_marginal_convergence2(samplesF,samplesG,F,G,M,node = 4)

