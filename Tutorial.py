import PGM
import scipy
# Tutorial on how to use PGM module
# It has 3 classes- factors, FactorList and CliqueTree. Let's start with 'factor'
#
# 'factor' class is used to replicate factors. The code
#
f = PGM.factor( var=[2,0,1], card= [2,2,2], val = scipy.ones(8) )
#
# creates a factor over variables X_2, X_0, X_1, which are all binary
# valued, because f.card[0] (the cardinality of X_2) is 2, 
# and likewise for X_0 and X_1. f has been initialized so that 
# f(X_2, X_0, X_1) = 1 for any assignment to the variables.
#
# A factor's values are stored in a row vector in the .val field 
# using an ordering such that the left-most variables as defined in the 
# .var field cycle through their values the fastest. More concretely, for 
# the factor phi defined above, we have the following mapping from variable 
# assignments to the index of the row vector in the .val field:
#
# -+-----+-----+-----+-----------------+   
#  | X_2 | X_0 | X_1 | f(X_2, X_0, X_1)|
# -+-----+-----+-----+-----------------+
#  |  0  |  0  |  0  |     f.val[0]    |
# -+-----+-----+-----+-----------------+
#  |  1  |  0  |  0  |     f.val[1]    |
# -+-----+-----+-----+-----------------+
#  |  0  |  1  |  0  |     f.val[2]    |
# -+-----+-----+-----+-----------------+
#  |  1  |  1  |  0  |     f.val[3]    |
# -+-----+-----+-----+-----------------+
#  |  0  |  0  |  1  |     f.val[4]    |
# -+-----+-----+-----+-----------------+
#  |  1  |  0  |  1  |     f.val[5]    |
# -+-----+-----+-----+-----------------+
#  |  0  |  1  |  1  |     f.val[6]    |
# -+-----+-----+-----+-----------------+
#  |  1  |  1  |  1  |     f.val[7]    |
# -+-----+-----+-----+-----------------+
#
#
# We have provided the A2I and I2A functions that compute the mapping between
# the assignments A and the variable indices I, given C, the cardinality of
# the variables. Concretely, given a factor phi, if f.val(I) corresponds to
# the assignment A, i.e. f(X = A) = f.val(I) then
# 
#   I = PGM.A2I(A, C)
#   A = PGM.I2A(I, C)
#
# For instance, for the factor phi as defined above, with the assignment 
#
#    A = [1,0,1] 
#
# to X_2, X_0 and X_1 respectively (as defined by phi.var = [3 1 2]), I = 5 
# as phi.val(6) corresponds to the value of phi(X_3 = 2, X_1 = 1, X_2 = 2).
# Thus, AssignmentToIndex([2 1 2], [2 2 2]) returns 6, and conversely, 
# IndexToAssignment(6, [2 2 2]) returns the vector [2 1 2]. The second
# argument in the function calls corresponds to the cardinality of the
# sample factor phi, phi.card, which is [2 2 2].
#
# More generally, the assignment vector A is a row vector that corresponds
# to assignments to the variables in a factor, with an understanding that the
# variables for which the assignments refer to are given by the .var field
# of the factor. 
#
# Giving A2I a matrix A, one assignment per row, will cause it
# to return a vector of indices I, such that I(k) is the index
# corresponding to the assignment in A(k, :) (row k). 
# 
# Similarly, giving I2A a vector I of indices will yield a
# matrix A of assignments, one per row, such that A(k, :) (the kth row of A)
# corresponds to the assignment mapped to by index I(k).
