import networkx as nx
import scipy
import itertools
import time
#from scipy.sparse import csr_matrix

def EliminateVar(n, g,clq_ind,E):
    """Eliminates n from graph defined by factorlist F"""
    l = len(clq_ind)
    new_clique = g.neighbors(n) # we will add 'n' to it later
    new_ind = scipy.array(g.neighbors(n))
    new_clique.append(n)
    
#    tri_g.add_edges_from( itertools.combinations(new_clique,2) )
    g.add_edges_from( itertools.combinations(new_clique,2) )    
    
    for i,clq in enumerate(clq_ind):
        if n in clq:
            E[l,i] = E[i,l] = 1
            clq_ind[i] = scipy.setdiff1d(clq,new_clique)
   
    clq_ind.append(new_ind)
    g.remove_node(n)

    return new_clique,g, clq_ind,E

def EliminateVar2(n, g,clq_ind,tree):
    """Eliminates n from graph defined by factorlist F"""
    l = len(clq_ind)
    new_clique = g.neighbors(n) # we will add 'n' to it later
    new_ind = scipy.array(g.neighbors(n))
    new_clique.append(n)
    
#    tri_g.add_edges_from( itertools.combinations(new_clique,2) )
    g.add_edges_from( itertools.combinations(new_clique,2) )    
    
    for i,clq in enumerate(clq_ind):
        if n in clq:
            tree.add_edge(i,l)
            clq_ind[i] = scipy.setdiff1d(clq,new_clique)
    tree.node[l]['clique'] = new_clique
    clq_ind.append(new_ind)
    g.remove_node(n)

#    return g, clq_ind,tree

def PruneTree(nodeList,E):
    to_remove = []
    for i,n in enumerate(nodeList):
        if i in to_remove:
            continue
        nbrs, = E[i].nonzero()
        for j in nbrs:
            if j in to_remove:
                continue
            if scipy.all([dummy in nodeList[j] for dummy in n])==True:
                for n2 in nbrs:
                    if n2 !=j:
                        E[j,n2] = 1
                        E[n2,j] = 1
                to_remove.append(i)
                E[i,:] = 0
                E[:,i] = 0
                break
    E=scipy.delete(E,to_remove,0) # delete rows and columns from adjacency matrix
    E=scipy.delete(E,to_remove,1)
    
    new_nodeList = [n for i,n in enumerate(nodeList) if i not in to_remove]
    return new_nodeList,E

def PruneTree2(tree):
    to_remove = []
    nodes = tree.nodes()
    for i in nodes:
        n = tree.node[i]['clique']
        if i in to_remove:
            continue
        nbrs = tree.neighbors(i)
        for j in nbrs:
            if j in to_remove:
                continue
            if scipy.all([dummy in tree.node[j]['clique'] for dummy in n])==True:
                for n2 in nbrs:
                    if n2 !=j:
                        tree.add_edge(j,n2)
                to_remove.append(i)
                tree.remove_node(i)
                break
#    E=scipy.delete(E,to_remove,0) # delete rows and columns from adjacency matrix
#    E=scipy.delete(E,to_remove,1)
    
#    new_nodeList = [n for i,n in enumerate(nodeList) if i not in to_remove]
    return tree

def min_neighbor_node(g):
    return min(g.degree_iter(),key = lambda item:item[1])[0]

def fill_edges(g,n):
    ngbrs = g.neighbors(n)
    # e = number of edges between neighbors of 'n'
    e = sum([g.edge[i].has_key(j) for i,j in itertools.combinations(ngbrs,2) ])
    return len(ngbrs)*(len(ngbrs)-1)/2 - e
def min_fill_node(g):    
    return min(g.nodes(), key = lambda n:fill_edges(g,n))    


def clique_tree(actual_g,min_fill = 0, order = []):
    """finds a clique tree of actual_g. min_fill=0 means min_neighbor elimination
       order: starting elimination order."""
    N = actual_g.number_of_nodes()    
    g = actual_g.copy()
    
    tri_g = nx.Graph()
    tri_g.add_nodes_from( g.nodes() )
    
    nodeList = []
    clq_ind = [] # For each clique, a list of nodes whose elimination would lead to each from that clique 
    E = scipy.zeros([N,N])
    tree = nx.Graph()
    tree.add_nodes_from(range(N))
    elim_order=[]
#    for n in order:
#        clique,g,clq_ind,E = EliminateVar(n, g,clq_ind,E)
##        clique,g,clq_ind,tree = EliminateVar2(n, g,clq_ind,tree)
#        nodeList.append(clique)
#        elim_order.append(n)
#        print clq_ind
#        
    for k in range( N-len(order) ):
        
        n = min_neighbor_node(g) if min_fill == 0 else min_fill_node(g)
        elim_order.append(n)
        EliminateVar2(n, g,clq_ind,tree)
    pot_edges=[]
    for i,j in itertools.combinations(actual_g.nodes(),2):
        if E[i,j]==1:
            pot_edges.append( (elim_order[i],elim_order[j]) )
    tree= PruneTree2(tree)
    clique_size = [len(data['clique']) for n,data in tree.nodes_iter(data=True)]
    tw = max(clique_size) - 1 # treewidth = (size of max clique) - 1    
    return tree,tw,elim_order,pot_edges
    
def get_triangulated_graph(nodeList):
    """returns a triangulated graph from nodeList(list of cliques)"""
    tri_g = nx.Graph()
    for clique in nodeList:
        tri_g.add_edges_from( itertools.combinations(clique,2) )
    return tri_g

if __name__ == '__main__':
    actual_g = nx.erdos_renyi_graph(10,0.3,seed = 1234)
    
#    n = 28
#    actual_g = nx.Graph()
#    actual_g.add_nodes_from(range(n**2))
#    for i in range(n):
#        for j in range(n-1):
#            actual_g.add_edge(n*i+j,n*i+j+1)
#    for i in range(n-1):
#        for j in range(n):
#            actual_g.add_edge(n*i+j,n*(i+1)+j)
            
    order = []#[9,8]
#    nodeList1,E1,tw1,elim_order1,pot_edges = clique_tree(actual_g,min_fill = 0)
#    tri1 = get_triangulated_graph(nodeList1)
    tree2,tw2,elim_order2,pot_edges = clique_tree(actual_g,min_fill = 1)
#    nodeList3,E3,tw3,elim_order3,pot_edges = clique_tree(actual_g,order=order)#scipy.random.permutation(10))
    
    print tw2