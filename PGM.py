# TO-DO: Make F connected, if not

import scipy
import networkx as nx
import itertools
import time
import scipy.ndimage
#import copy

class factor(object):
    def __init__(self,var,card,val):
        self.var =  scipy.array(var)
        self.card = scipy.array(card).astype(int)
        self.val =  scipy.array(val)
        
    def FactorOperation(self,f,oper='P'):
        """Takes factor product with another factor f
           oper ='P' for Product,'S' for Sum """
        if len(self.var) == 0:
            return f
        if len(f.var) == 0:
            return self
        var = scipy.union1d(self.var,f.var)
        card = scipy.empty(len(var))
        
        map1 = [scipy.where(var==i)[0][0] for i in self.var]
        map2 = [scipy.where(var==i)[0][0] for i in f.var]
        card[map1]=self.card
        card[map2] = f.card
        
        val = scipy.zeros(card.prod(dtype=int))
        
        assignments = I2A(range(int( card.prod() )), card)
        indx1 = A2I(assignments[:, map1], self.card)
        indx2 = A2I(assignments[:, map2], f.card)
        if oper == 'P':
            val = self.val[indx1] * f.val[indx2]
        elif oper =='S':
            val = self.val[indx1] + f.val[indx2]
        return factor(var,card,val)

    def ObserveEvidence(self,E):
        """Return the factor corresponding to evidence E,
        E is a dictionary with variable name as keys, and value observed as values"""
        for v,e in E.iteritems():
            indx, = (self.var==v).nonzero() # find i where f.var[i]=v
            if indx.size != 0:
                A = I2A(range(len(self.val)), self.card) # all assignments
                A = A[:,indx].flatten() # only interested in 'indx' element of each row
                self.val[A != e] = 0

    def Marginalize(self,V):
        """Eliminate Variables present in list V"""
        var = scipy.setdiff1d(self.var,V)
        map1 = [scipy.where(self.var==i)[0][0] for i in var]
        card = self.card[map1]
        
        assignments = I2A(range(len(self.val)), self.card)
        indx = A2I(assignments[:, map1], card)
        
        val = scipy.ndimage.sum(self.val,indx,index = range( card.prod() ))
        return factor(var,card,val)

    def MaxMarginalize(self,V):
        """Eliminate Variables present in list V"""
        var = scipy.setdiff1d(self.var,V)
        map1 = [scipy.where(self.var==i)[0][0] for i in var]
        card = self.card[map1]
        
        assignments = I2A(range(len(self.val)), self.card)
        indx = A2I(assignments[:, map1], card)
        
        val = scipy.ndimage.maximum(self.val,indx,index = range( card.prod() ))
        return factor(var,card,val)
        
    def copy(self):
        return factor(self.var,self.card,self.val)
        
    def __mul__(self,f):
        return self.FactorOperation(f,'P')
    def __add__(self,f):
        return self.FactorOperation(f,'S')
    def __pow__(self,p): # useful for TRW Message passing
        return factor(self.var,self.card,self.val**p)

class FactorList(object):
    """Contains a list of factors"""
    def __init__(self,factors):
        self.factors = list(factors)
        self.g = self.create_graph()
        self.cardVec = self.getCardVec()
        self.var2facs = self.getVar2FactorsMap()
    def update(self):
        """Updates the graph if some factors are added to the list"""
        self.g = self.create_graph()
    def create_graph(self):
        """Creates networkx graph"""
        g = nx.Graph()
        
        for f in self.factors:
            g.add_nodes_from( f.var )
            g.add_edges_from( itertools.combinations(f.var,2) )
        return g
    
    def getAllNodes(self):
        """returns all nodes"""
        return self.g.nodes()
    def getCardVec(self):
        """cardinality vector: cardVec[i] = cardinality of node i"""
        V = self.getAllNodes()
        cardVec = scipy.empty_like(V)
        for i,v in enumerate(V):
            for f in self.factors:
                if v in f.var:
                    index, = (f.var == v).nonzero()                
                    cardVec[i]=f.card[ index[0] ]
                    break
        return cardVec.astype(int)
        
    def getAdjMatrix(self):
        return nx.adj_matrix(self.g)

    def getVar2FactorsMap(self):
        """gives a list where each element represents a node and gives a list of 
           indices of factors in "F" which contain that node in its scope"""
        V = self.getAllNodes()
        return list(list(idx for idx,f in enumerate(self.factors) if i in f.var) for i in V) 

    def make_connected(self):
        """If g is not connected, add redundant factors to make it connected)"""
        if nx.is_connected(self.g): return
        import random
        cc = list( nx.connected_components(self.g) )
        nodes = [random.sample(cluster,1)[0] for cluster in cc]
        for n1,n2 in zip(nodes[:-1],nodes[1:]):
            self.factors.append(factor(var=[n1,n2], card=self.cardVec[[n1,n2]], val=scipy.ones(4)))
        self.update()
        
    def JointDistn(self):
        """Computes unnormalized Joint Distribution based on the list of factors"""
        if len(self.factors)==1:
            return self.factors[0]
        F = self.factors[0]
        for i in range(1,len(self.factors)):
            F = F * self.factors[i]
        self.JDist = F
        return F

    def ObserveEvidence(self,E):
        """Return the factor corresponding to evidence E,
        E is a dictionary with variable name as keys, and value observed as values"""
        for f in self.factors:
            f.ObserveEvidence(E)

    def MarginalDistn(self, V, E={} ):
        """Computes Marginal Distn of variable list V under evidence E"""
        
        J= self.JointDistn()
        J.ObserveEvidence(E)
        M = J.Marginalize(scipy.setdiff1d(J.var,V))
        M.val = M.val/(sum(M.val))
        return M
    
    def run_inference(self,isMax = 1,findZ = 0):
        """Runs Inference; isMax=1 for MAP,0 for Marginals;
           findZ = 1 to compute Z while performing Marginal Inference"""
#        st=time.time()
        self.make_connected()
        self.nop = 0 # number of operations
        T=CliqueTree(self,isMax,findZ)
        if isMax == 0:
            self.marg_clique_tree = T
        elif isMax==1:
            self.MAP_clique_tree = T
#        print time.time()-st'=
        self.nop += T.nop
        M=[]
        for i in self.g.nodes(): # assuming nodes are labeled 0..N-1
            for s,data in T.nodes_iter(data=True):
                f = data['fac']
                if i in f.var:
                    if isMax==0:
                        dummy = f.Marginalize(scipy.setdiff1d(f.var,i))
                        if findZ == 0:
                            dummy.val = dummy.val/sum(dummy.val)
                    else:
                        dummy = f.MaxMarginalize(scipy.setdiff1d(f.var,i))
                    self.nop += scipy.prod(f.card)
                    M.append(dummy)
                    break
#        print time.time()-st
        return M
    
    def find_clique_marginal(self,cliques):
        """Takes a list of cliques as an input; output is a list of marginals
           of tose cliques. Assumes inference has been run on F, so it has an
           entity F.marg_clique_tree"""
        M = []   
        for clq in cliques:
            for f in self.marg_clique_tree.factors:                
                if scipy.all([s in f.var for s in clq]):
                    marg = f.Marginalize(scipy.setdiff1d(f.var,clq))
                    marg.val = marg.val/sum(marg.val)
                    M.append(marg)
                    break
        return M
        
class CliqueTree(nx.DiGraph):
    """Clique Tree object: it is "directed tree" which means self.to_undirected()
    is an undirected tree. Each node contains calibrated factors and each directed
    edge contains message passed from O to D of that edge"""
    def __init__(self,F,isMax,findZ):
        """initializes the clique tree from a FactorList"""
        nx.DiGraph.__init__(self)
        tree = create_clique_tree(F.g)
        d_tree = tree.to_directed()
        self.add_nodes_from( d_tree.nodes(data=True) )
        self.add_edges_from( d_tree.edges(data=True) )
        self.nop = 0
        
        self.compute_clique_potentials(F)
        
        self.calibrate(isMax,findZ)

    def compute_clique_potentials(self,F):
        """Computes initial potentials for clique trees"""
        N=self.number_of_nodes()
        
        #assignment of factors to cliques
        alpha = -1*scipy.ones(len(F.factors), dtype=int)
        
        for i,f in enumerate(F.factors):
            for j,data in self.nodes_iter(data=True):
                if len(scipy.setdiff1d(f.var,data['clique']) ) ==0:
                    alpha[i] = int(j)
                    break
        for i in range(N):
            var = scipy.array(self.node[i]['clique'],dtype=int)
            card = F.cardVec[var]
            val = scipy.ones( card.prod() )
            self.node[i]['fac'] = factor(var,card,val)
        
        for i,j in enumerate(alpha):
            self.node[j]['fac'] *= F.factors[i]
            self.nop += scipy.prod(self.node[j]['fac'].card)
        
    def calibrate(self,isMax,findZ):
        
        N = self.number_of_nodes()
        if isMax==1:
            for i in self.nodes():
                self.node[i]['fac'].val=scipy.log(self.node[i]['fac'].val)
        
        for i,j in self.edges():
            self.edge[i][j]['msg'] = factor([],[],[])
            self.edge[i][j]['msg_ind'] = 0# message passed from i to j or not
        
        I,J = get_next_cliques(self)
        while I >= 0:
            dummy = self.node[I]['fac']
            for k in self.predecessors(I):
                if self.edge[k][I]['msg_ind']==1 and k !=J:# temp change, might switch back later
                    if isMax==0:
                        dummy *= self.edge[k][I]['msg']
                    else:
                        dummy += self.edge[k][I]['msg']
                    self.nop += scipy.prod(dummy.card)
            if isMax==0:
                self.edge[I][J]['msg']= dummy.Marginalize( scipy.setdiff1d(self.node[I]['fac'].var,self.node[J]['fac'].var))
                if findZ==0: self.edge[I][J]['msg'].val=self.edge[I][J]['msg'].val/sum(self.edge[I][J]['msg'].val)
            else:
                self.edge[I][J]['msg']=dummy.MaxMarginalize( scipy.setdiff1d(self.node[I]['fac'].var,self.node[J]['fac'].var))
            self.nop += scipy.prod(dummy.card)

            self.edge[I][J]['msg_ind'] = 1 # message passed from I to J
            I,J = get_next_cliques(self)
            
        for i in self.nodes():
            if isMax==0:
                self.node[i]['fac'] *= reduce(lambda x,y:x*y,(self.edge[j][i]['msg'] for j in self.successors(i)))
            else:
                self.node[i]['fac'] += reduce(lambda x,y:x+y,(self.edge[j][i]['msg'] for j in self.successors(i)))
            self.nop += N*scipy.prod(self.node[i]['fac'].card) # check this

def min_fill_node(g):
    """returns the node with minimum fill edges"""
    return min( g.nodes(),key = lambda x:fill_edges(g,x) )
def fill_edges(g,n):         
    ngbrs = g.neighbors(n)        
    # e = number of edges between neighbors of 'n' 
    e = sum([g.edge[i].has_key(j) for i,j in itertools.combinations(ngbrs,2) ])
    return len(ngbrs)*(len(ngbrs)-1)/2 - e

def create_clique_tree(g):
    """creates clique tree from undirected graph g; and changes it to """
    
    N = g.number_of_nodes()        
    g2 = g.copy()
    
    clq_ind = []# For each clique, a list of nodes whose elimination would lead to each from that clique
    tree = nx.Graph()
    tree.add_nodes_from(range(N))
    for k in range(N):
#            sorted_list = sorted(g2.degree_iter(),key = lambda item:item[1]) # sort by degree
#            n = sorted_list[0][0] # extract first element (min neighbors)
        n = min_fill_node(g2) # uncomment above 2 lines for min-neighbor
        eliminate_var(n, g2,clq_ind,tree)
    tree = prune_tree(tree)
    return tree.to_directed()
        
def eliminate_var(n, g,clq_ind,tree):
    """Eliminates n from graph g; updates cld_ind and tree"""
    l = len(clq_ind)
    new_clique = g.neighbors(n) # we will add 'n' to it later
    new_ind = scipy.array(g.neighbors(n))
    new_clique.append(n)
    
    g.add_edges_from( itertools.combinations(new_clique,2) )    
    
    for i,clq in enumerate(clq_ind):
        if n in clq:
            tree.add_edge(l,i)
            clq_ind[i] = scipy.setdiff1d(clq,new_clique)
    
    clq_ind.append(new_ind)
    g.remove_node(n)
    tree.node[l]['clique'] = new_clique

def prune_tree(tree):
    nodes = tree.nodes() # copy since tree.nodes() will be modified
    for i in nodes:
        nbrs = tree.neighbors(i)
        for j in nbrs:
            # if i \subset j
            if len(scipy.setdiff1d(tree.node[i]['clique'],tree.node[j]['clique'])) == 0:                
                tree.add_edges_from([(j,n2) for n2 in nbrs if n2 != j])
                tree.remove_node(i)
                break
    return tree
        
def get_next_cliques(tree):
    """outputs next pair of cliques between whom message can be passed,
    If negative numbers, no pair of cliques possible"""    
    for i in tree.nodes():
        for j in tree.successors(i):
            if tree.edge[i][j]['msg_ind']==0: #no message from i to j
            # if all neighbouring cliques except j have sent a message
                msg_indices = [tree.edge[k][i]['msg_ind'] for k in tree.predecessors(i) if k!=j]
                if scipy.all(msg_indices):
                    return i,j
    return -1,-1 # returning -1 would mean no more cliques; all messages have been passed
    
def max_decode(M):
    asgnmnt = scipy.array([ f.val.argmax() for f in M])
    return asgnmnt
    
def A2I(A,card):
    """Takes assignment and cardinality vector as an input, outputs its index in
       the val array"""
    A = scipy.atleast_2d(A)
    return scipy.sum( ( scipy.hstack([1,card[:-1]]).cumprod() )* A ,1).astype(int)

def I2A(I,card):
    """Takes  array index and cardinality vector as an input, returns assignment"""
    m,n = len(I), len(card)
    A = (scipy.repeat(I,n)//scipy.tile( scipy.hstack([1,card[:-1]]).cumprod(), m)) % scipy.tile(card, m)
         # // - floor division, % - modulus
    return A.reshape([m,n])
    
def factors2ExpFam(F):
    F.make_connected()
    m = len(F.g.nodes())
    theta = scipy.zeros([m,m])

    I = scipy.eye(m)
    A = scipy.vstack( (I,scipy.zeros(m)) )
    val_s = scipy.sum( scipy.log([f.val[A2I(A[:,f.var],f.card)] for f in F.factors]), axis=0)
    scipy.fill_diagonal(theta, val_s[:-1]-val_s[-1]) # theta_s[s] = f1[s]/f0 goes in diagonal
    
    A = scipy.array( [I[s]+I[t] for s,t in F.g.edges()])
    val = scipy.sum(scipy.log([f.val[A2I(A[:,f.var],f.card)] for f in F.factors]), axis=0)
    for i, (s,t) in enumerate(F.g.edges()):
        if s>t: s,t = t,s # s should be smaller than t            
        theta[s,t] = val[i]+val_s[-1]-val_s[s]-val_s[t] #f11*f00/f01*f10
    return theta
    
def expfam2Factors(theta):
    """Convert from theta to factorList"""
    facList = []
    m = theta.shape[0]
    for s,th in enumerate(scipy.diag(theta)):
        facList.append(factor(var=[s],card=[2],val = [1,scipy.exp(th)]))
    for s,t in itertools.combinations(range(m),2):
        if not scipy.isclose(theta[s,t],0):
            facList.append(factor(var=[s,t],card=[2,2],val=[1,1,1,scipy.exp(theta[s,t])]))
    F = FactorList(facList)
    F.make_connected()
    return F

def test_conversion():
    """Test function for factors2exp and expfam2Factors; converts F->expFam->F2
       prints True if F and F2 have same distribution; which is necessary for
       functions to work. Something wrong if False is printed"""
    f1 = factor([0,1],[2,2],scipy.rand(4))
    f2 = factor([1,2],[2,2],scipy.rand(4))
    f3 = factor([3],[2],scipy.rand(2))

    F = FactorList([f1,f2,f3])
    theta = factors2ExpFam(F)
    F2 = expfam2Factors(theta)
    ratio = F2.JointDistn().val/ (F.JointDistn().val)
    ratio = ratio/ratio[0]
    print scipy.allclose(ratio,1)

def greedy_MAP_assignment(theta,random_runs = 10,heur = 'first'):
    """starts with a random assignment; then make greedy improvements by
       looking at neighboring assignments until possible;
       random_runs: Number of random restarts, best of all is picked, default=10
       heur:'first' changes the first node with improvement,'best' looks at all
              nodes and changes the one with maximum improvement"""
    N = theta.shape[0]
    scipy.random.seed()
    max_p = -scipy.inf
    for k in range(random_runs):
        A = scipy.random.randint(2,size = N)
        improved = True
        p = A.dot( theta.dot(A) )
        while improved:
            improved = False
            if heur == 'first':
                p2 = -scipy.inf
                perm = scipy.random.permutation(N)
                for s in perm:
                    #dp: change in p if A[i] bit is reversed
                    dp = (1-2*A[s])*( A.dot(theta[s,:]+ theta[:,s]) ) + theta[s,s]
                    if dp>0:
                        p2 = dp
                        break

            if heur == 'best':
                dp = (1-2*A)*( A.dot(theta + theta.T) ) + scipy.diag(theta)
                p2,s = dp.max(), dp.argmax()
            if p2 > 0:
                A[s] = 1-A[s]
                improved = True
                p += p2
        if p>max_p:
            greedy_A,max_p = A.copy(),p
    return greedy_A.astype(int),max_p