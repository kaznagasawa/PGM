# This file tests MAP inference
# When executed, it should print out 'mornings'

import PGM
import scipy
import scipy.io
import time
import networkx as nx
import cProfile

#data = scipy.io.loadmat('GetNextC.mat')
#edges = data['edges']
#ind = data['I'] # message passed indicator, not symmetric
#N = edges.shape[0]
#data2 = scipy.io.loadmat('calibrate.mat')
pr=cProfile.Profile()


data2 = scipy.io.loadmat('MAP_test.mat')
val_array = data2['val_array'].astype(float)
card_array = data2['card_array']
var_array = data2['var_array'].astype(int)

factors = []
nodeList = []
for i in range(21):
    var = var_array[i][var_array[i]<20]
    var = var - 1
    card = card_array[i][card_array[i]<30]
    val = val_array[i][val_array[i] < 20]
    factors.append(PGM.factor(var,card,val))
    nodeList.append(var)

F=PGM.FactorList(factors)
isMax = 0

#g = nx.erdos_renyi_graph(10,.3,seed=1234) # g should be connected
#scipy.random.seed(38422)
#fac_list = []
#for e1,e2 in g.edges():
#    f = PGM.factor(var=[e1,e2],card=[2,2],val=scipy.rand(4))
#    f.val[3] = (1+.4*scipy.rand()) * f.val[1] * f.val[2] / f.val[0]
#    fac_list.append(f)
#F = PGM.FactorList(fac_list)
    
st = time.time()

pr.enable()
M = F.run_inference(isMax)
pr.disable()

A = PGM.max_decode(M) 
print time.time()-st

import string
d = dict(enumerate(string.ascii_lowercase))
output=' '
for a in A:
    output=output+d[a]
print output

pr.print_stats(sort='tottime')
#----Copied from test.py.
#---code for testing PGM functions

#f1 = PGM.factor([0],[2],[0.11,0.89])
#f2 = PGM.factor([1,0],[2,2],[0.59,0.41,0.22,0.78])
#f3 = PGM.factor([2,1],[2,2],[0.39,0.61,0.06,0.94])
#F = PGM.FactorList([f1,f2,f3])
#
#V= [1]
#E = {1:0,2:1}
#FP=f1.FactorProduct(f2)

#FP = f1*f2*f3
#FS = f1+f2

#G=F.JointDistn()

#M= f2.Marginalize([1])

#F.ObserveEvidence(E)
#M=F.MarginalDistn([1,2],{0:1})
#M.val = M.val/sum(M.val)

#data = scipy.io.loadmat('PA4Sample.mat')