#!/usr/bin/env python
# coding: utf-8

# ### A matrix is rectangular array of numbers or functions which is enclosed in brackets.

# In[2]:


## A =[1 2 3]...
## example of 3x3 matric as below
import numpy as np

mat1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
mat1


# ### Addition of two matrices

# In[3]:


## Addition of two matrices 
## matrices must be same in size
## Each element of mtrix added with corresponding element
## following rules can be satisfy
## if matrx A and B are in same size than 1. A+B =B + A, 2. c(A+B) = cA+cB, 3. (A+B)+C = A+(B+C), 4. (c+k)A = cA+kA
## 5. A+0 = A, 6. c(kA) = (ck)A, 7. A+(-A) = 0, 8. 1A = A
## let example A=[[1,2,3],[3,2,1],[7,8,9]]  and B = [[2,3,1],[1,4,1],[1,5,7]]

A = np.array([[1,2,3],[3,2,1],[7,8,9]])

B = np.array([[2,3,1],[1,4,1],[1,5,7]])

C = np.array([[2,5,6],[3,7,6],[1,2,6]])

print(A)
print()
print(B)


# In[4]:


## proof for 1. A+B = B+A

AaddB = A + B
BaddA = B + A
 
print(AaddB == BaddA)  
print()
print(AaddB)
print()
print(BaddA)


# In[5]:


## proof for 5. A+0 = A
MAT_ZERO = A+0

print(MAT_ZERO)


# In[6]:


## proof for 3. (A+B)+C = A+(B+C)

Mat1abc = (A + B) + C
Mat2abc = A + (B + C)

print(Mat1abc == Mat2abc)
print()
print(Mat1abc)
print()
print(Mat2abc)


# ### Multiplication of two matrices

# In[7]:


## if two matrices Amxn and Bjxk for multiplication m must be equel to k(m=k) than only multiplication perform.
## let example A=[[1,2,3],[3,2,1],[7,8,9]]  and B = [[2,3,1],[1,4,1],[1,5,7]]
## 1. AB != BA

A = np.array([[1,2,3],[3,2,1],[7,8,9]])

B = np.array([[2,3,1],[1,4,1],[1,5,7]])

C = np.array([[2,5,6],[3,7,6],[1,2,6]])


Mul_AB = A.dot(B)

print(Mul_AB)


# In[8]:


# proof for 1. AB != BA
## multilpication of AB is not equel multiplcation of BA
Mul_AB = A.dot(B)
Mul_BA = B.dot(A)

print("A Multiply by B is \n",Mul_AB)
print()
print("B Multiply by A is \n",Mul_BA)


# ### Transpose of A matrix

# In[9]:


## Transpose of matrix means either rows interchange with columns or viceversa
## Suppose we have a matrix A as below
A = np.array([[1,2,3],[3,2,1],[7,8,9]])


# In[10]:


TranOfA = np.transpose(A)
print("Mat A =\n",A)
print()
print("Transpose of Mat A =\n",TranOfA)


# ### Symmetric matrix

# In[11]:


## A matrix which is equalt it's transpose. matrix A = transpose of A.
## Example A=[[1,7,3],[7,4,-5],[3,-5,6]] is symmetric matrix
## proof for A matrix is symmetric matrix

A = np.array([[1,7,3],[7,4,-5],[3,-5,6]])

TransOfA = np.transpose(A)

print(A)
print()
print(TransOfA)

print()
print(A == TransOfA)


# ## Skew symmetric matrix

# In[12]:


## A matrix is called skew symmetric matrix if Matrix A = -(transpose of A)
## Example A = ([[0,2,-45],[-2,0,-4],[45,4,0]])
## proof for A skew smmetric matrix

A = np.array([[0,2,-45],[-2,0,-4],[45,4,0]])

TransposeOfA = np.transpose(A)

print(A)
print()
print(TransposeOfA)
print()
print(A == -(TransposeOfA))


# ### Determinant of a Matrix 

# In[13]:


##The determinant of a matrix is a special number that can be calculated from a square matrix.
##It is used to find the inverse of matrix and system of a linear equation

A=[[1,3],[2,1]]

DeterminatOfA = np.linalg.det(A)

print(DeterminatOfA)
print()
print(int(DeterminatOfA))


# In[14]:


A = np.array([[4, 3, 2], [-2, 2, 3], [3, -5, 2]])
B = np.array([25, -10, -4])
X2 = np.linalg.solve(A,B)

print(X2)


# ### Cramer’s Rule
# 

# In[15]:


##Cramer’s Rule uses determinants to solve for a solution to the equation Ax=b, when A is a square matrix
##Use Cramer’s Rule to solve for a single variable in a system of linear equation.
##Example 3x1 + x2 = 4 and 2x1 + x2 = 3 find the solution for x1 and x2.

## matrix A = [[3,1],[2,1]] and B = [[4,3]]

A = np.array([[3,1],[2,1]])
B = np.array([4,3])


# In[16]:


sol = np.linalg.solve(A,B)

print(sol)


# In[17]:


## Next example 20x + 10y = 350 and 17x + 22y = 500

A = np.array([[20, 10], [17, 22]])
B = np.array([350, 500])
X = np.linalg.solve(A,B)

print(X)


# ### Inverse of a Matrx

# In[18]:


##matrix don't have divide operation so same we have to use inverse
## A.invers(A) = I

A = [[40,70],[20,60]]

MatA = np.array(A)

InvA = np.linalg.inv(A)

print(InvA)


# In[19]:


## proof A.inverse(A) = I, I is unit or identity matrix

I = MatA.dot(InvA)

print(I)


# In[20]:


x = np.linalg.matrix_rank(I)

print(x)


# ### Rank Of a Matrix

# The rank of a matrix is defined as 
# (a) the maximum number of linearly independent column vectors in the matrix or 
# (b) the maximum number of linearly independent row vectors in the matrix. Both definitions are equivalent.
# 
# For an r x c matrix,
# 
# If r is less than c, then the maximum rank of the matrix is r.
# If r is greater than c, then the maximum rank of the matrix is c.
# The rank of a matrix would be zero only if the matrix had no elements. If a matrix had even one element, its minimum rank would be one.

# In[21]:


A = [[1,2,3],[4,6,7],[1,0,1]]
rankOfA = np.linalg.matrix_rank(A)

print("Rank of matrix A is : ", rankOfA)


# ### Homogeneous systems 

# 
# 
# A system of linear equations is homogeneous if all of the constant terms are zero:
# 
# ex. 2X+5Y = 0, 4X-8Y = 0.
# 
# 
# Homogeneous solution set
# Every homogeneous system has at least one solution, known as the zero (or trivial) solution, which is obtained by assigning the value of zero to each of the variables. If the system has a non-singular matrix (det(A) ≠ 0) then it is also the only solution. If the system has a singular matrix then there is a solution set with an infinite number of solutions. This solution set has the following additional properties:
# 
# If u and v are two vectors representing solutions to a homogeneous system, then the vector sum u + v is also a solution to the system.
# If u is a vector representing a solution to a homogeneous system, and r is any scalar, then ru is also a solution to the system.
# These are exactly the properties required for the solution set to be a linear subspace of Rn. In particular, the solution set to a homogeneous system is the same as the null space of the corresponding matrix A. Numerical solutions to a homogeneous system can be found with a singular value decomposition.

# In[22]:


##for above example A = [[2,5],[4,-8]], B = [0,0]
A = [[2,5,4],[4,-8,1],[3,5,1]]
B = [0,0,0]

solution_of_A = np.linalg.solve(A,B)

print(solution_of_A)


# In[23]:


## Example x-y = 0, 2x+y = 0

A =[[1,-1],[2,1]]
B =[0,0]

solutionOfA = np.linalg.solve(A,B)

print(solutionOfA)


# In[24]:


MAT1 = [[-1,1,4],[1,3,8],[0.5,1,2.5]]
MAT2 = [0,0,0]

#sol = np.linalg.solve(MAT1,MAT)

det = np.linalg.det(MAT1)
## when homogeneous matrix having determinat zero itmeans system having infinite solution.
print(det)


# ## Home work solution

# In[31]:


get_ipython().run_cell_magic('time', '', '#problem - 2\n##solve in three steps 2, 4, 28 \nMatA= np.array([[1,2],[3,2]])\n#square of matrix\nmat = MatA.dot(MatA)\n#power 4 of matrix\nmat4 = mat.dot(mat)\n#power 28 of matrix\nmat28 = np.linalg.matrix_power(mat4,7)\n\nprint(mat28)')


# In[32]:


##solve in 1- step but it's taking more time than above solution
%%time
MatA= np.array([[1,2],[3,2]])
mul28 = np.linalg.matrix_power(MatA,28)

print(mul28)


# In[33]:


##check the solution's value
print(mat28 == mul28)


# In[96]:


#problem - 1

ran = [[1,0],[2,1],[1,0]]

rankOF = np.linalg.matrix_rank(ran)

print(rankOF)


# In[ ]:




