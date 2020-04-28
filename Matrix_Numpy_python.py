#!/usr/bin/env python
# coding: utf-8

# ### A matrix is rectangular array of numbers or functions which is enclosed in brackets.

# In[2]:


## A =[1 2 3]...
## examole of 3x3 matric as below
import numpy as np

mat1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
mat1


# ### Addition of two matrices

# In[21]:


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


# In[13]:


## proof for 1. A+B = B+A

AaddB = A + B
BaddA = B + A
 
print(AaddB == BaddA)  
print()
print(AaddB)
print()
print(BaddA)


# In[20]:


## proof for 5. A+0 = A
MAT_ZERO = A+0

print(MAT_ZERO)


# In[26]:


## proof for 3. (A+B)+C = A+(B+C)

Mat1abc = (A + B) + C
Mat2abc = A + (B + C)

print(Mat1abc == Mat2abc)
print()
print(Mat1abc)
print()
print(Mat2abc)


# ### Multiplication of two matrices

# In[28]:


## if two matrices Amxn and Bjxk for multiplication m must be equel to k(m=k) than only multiplication perform.
## let example A=[[1,2,3],[3,2,1],[7,8,9]]  and B = [[2,3,1],[1,4,1],[1,5,7]]
## 1. AB != BA

A = np.array([[1,2,3],[3,2,1],[7,8,9]])

B = np.array([[2,3,1],[1,4,1],[1,5,7]])

C = np.array([[2,5,6],[3,7,6],[1,2,6]])


Mul_AB = A.dot(B)

print(Mul_AB)


# In[39]:


# proof for 1. AB != BA
## multilpication of AB is not equel multiplcation of BA
Mul_AB = A.dot(B)
Mul_BA = B.dot(A)

print("A Multiply by B is \n",Mul_AB)
print()
print("B Multiply by A is \n",Mul_BA)


# ### Transpose of A matrix

# In[34]:


## Transpose of matrix means either rows interchange with columns or viceversa
## Suppose we have a matrix A as below
A = np.array([[1,2,3],[3,2,1],[7,8,9]])


# In[40]:


TranOfA = np.transpose(A)
print("Mat A =\n",A)
print()
print("Transpose of Mat A =\n",TranOfA)


# ### Symmetric matrix

# In[45]:


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

# In[48]:


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

# In[52]:


##The determinant of a matrix is a special number that can be calculated from a square matrix.
##It is used to find the inverse of matrix and system of a linear equation

A=[[1,3],[2,1]]

DeterminatOfA = np.linalg.det(A)

print(DeterminatOfA)
print()
print(int(DeterminatOfA))


# In[53]:


A = np.array([[4, 3, 2], [-2, 2, 3], [3, -5, 2]])
B = np.array([25, -10, -4])
X2 = np.linalg.solve(A,B)

print(X2)


# ### Cramer’s Rule
# 

# In[58]:


##Cramer’s Rule uses determinants to solve for a solution to the equation Ax=b, when A is a square matrix
##Use Cramer’s Rule to solve for a single variable in a system of linear equation.
##Example 3x1 + x2 = 4 and 2x1 + x2 = 3 find the solution for x1 and x2.

## matrix A = [[3,1],[2,1]] and B = [[4,3]]

A = np.array([[3,1],[2,1]])
B = np.array([4,3])


# In[59]:


sol = np.linalg.solve(A,B)

print(sol)


# In[60]:


## Next example 20x + 10y = 350 and 17x + 22y = 500

A = np.array([[20, 10], [17, 22]])
B = np.array([350, 500])
X = np.linalg.solve(A,B)

print(X)


# In[ ]:




