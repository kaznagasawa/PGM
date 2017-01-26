import networkx as nx
import scipy
import itertools

def EliminateVar(n, g,clq_ind,tree):
    """Eliminates n from graph defined by factorlist F"""
    l = len(clq_ind)
    new_clique = g.neighbors(n) # we will add 'n' to it later
    new_ind = scipy.array(g.neighbors(n))
    new_clique.append(n)
    
    g.add_edges_from( itertools.combinations(new_clique,2) )    
    
    for i,clq in enumerate(clq_ind):
        if n in clq:
            tree.add_edge(i,l)
            clq_ind[i] = scipy.setdiff1d(clq,new_clique)
    tree.node[l]['clique'] = new_clique
    clq_ind.append(new_ind)
    g.remove_node(n)


def PruneTree(tree):
    nodes = tree.nodes()
    for i in nodes:
        nbrs = tree.neighbors(i)
        for j in nbrs:
            if len(scipy.setdiff1d(tree.node[i]['clique'],tree.node[j]['clique'])) == 0:                
                tree.add_edges_from([(j,n2) for n2 in nbrs if n2 != j])
                tree.remove_node(i)
                break
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
    
    clq_ind = [] # For each clique, a list of nodes whose elimination would lead to each from that clique 
    tree = nx.Graph()
    tree.add_nodes_from(range(N))
    elim_order=[]
    for n in order:
        elim_order.append(n)
        EliminateVar(n, g,clq_ind,tree)
#        
    for k in range( N-len(order) ):
        n = min_neighbor_node(g) if min_fill == 0 else min_fill_node(g)
        elim_order.append(n)
        EliminateVar(n, g,clq_ind,tree)
    tree = PruneTree(tree)
    clique_size = [len(data['clique']) for n,data in tree.nodes_iter(data=True)]
    tw = max(clique_size) - 1 # treewidth = (size of max clique) - 1    
    return tree,tw,elim_order
    
def get_triangulated_graph(tree):
    """returns a triangulated graph from tree"""
    tri_g = nx.Graph()
    for i in tree.nodes():
        tri_g.add_edges_from( itertools.combinations(tree.node[i]['clique'],2) )
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
            
    order = [9,8]
    tree1,tw1,elim_order1 = clique_tree(actual_g,min_fill = 0)
    tri1 = get_triangulated_graph(tree1)
    tree2,tw2,elim_order2 = clique_tree(actual_g,min_fill = 1)
    tree3,tw3,elim_order3 = clique_tree(actual_g,order=order)#scipy.random.permutation(10))
    
    print tw1,tw2,tw3