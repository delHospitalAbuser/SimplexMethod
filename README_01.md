Implementation of a simplex algorithm based on optimization theory lectures I attended. 
Without going into the math, I will describe how the code works on example:

 ### Formulating the problem,  the system of equations should be transformed into a matrix form.

Maximize function $f=x+y+z$ with constraints:

$$\left\{\begin{array}{l}
   x-y-2z\leq 5\\
   x+2y-3z\leq 3\\
   x+2y+6z\leq 5\\
   2x+y-5z\leq 8\\
   x,y,z\geq 0\ .
  \end{array}\right.$$

 To get matrix form we need to add some variables to change inequalities into equalities, so let a,b,c,d such that:

$$\left\{\begin{array}{l}
   x-y-2z +a = 5\\
   x+2y-3z +b = 3\\
   x+2y+6z +c =  5\\
   2x+y-5z + d = 8
  \end{array}\right.$$

To get maximum of the function we have to change signs in the objective function. Then we get a minimalization problem which can be solved by the simplex method.   We will change the sign of the objective function in the solution to get its maximum. In this case, the matrix has the form:



```python
var('a','b','c','d','x','y','z','f')

M = matrix(
[
[a,b,c,d,x,y,z,f],
[0,0,0,0,-1,-1,-1,0],
[1,0,0,0,1,-1,-2,5],
[0,1,0,0,1,2,-3,3],
[0,0,1,0,1,2,6,5],
[0,0,0,1,2,1,-5,8],
]
)

M
```




    [ a  b  c  d  x  y  z  f]
    [ 0  0  0  0 -1 -1 -1  0]
    [ 1  0  0  0  1 -1 -2  5]
    [ 0  1  0  0  1  2 -3  3]
    [ 0  0  1  0  1  2  6  5]
    [ 0  0  0  1  2  1 -5  8]



### Now we need to calculate the simplex table. 
To do this, we need to find the basis vectors which are the ones that make up the identity matrix.


```python
base_values = []
for i in range(M.ncols()-1):
    if sum(map(abs, vector(M[2:,0:M.ncols()-1][:,i]))) == 1 and max(vector(M[2:,0:M.ncols()-1][:,i])) == 1:
        base_values.append(M[1,i])
base_values
```




    [0, 0, 0, 0]



We iterate through the columns of the matrix looking for vectors that may be an element of the identity matrix, the sum of the absolute values of the elements of a vector must be equal to 1, as well as its largest element, then the vector consists of exactly one 1 and the rest are zeros. If the vector meets the conditions, we need to get the corresponding values in the objective function.

To make it clearer, let's look at a matrix:


```python
M
```




    [ a  b  c  d  x  y  z  f]
    [ 0  0  0  0 -1 -1 -1  0]
    [ 1  0  0  0  1 -1 -2  5]
    [ 0  1  0  0  1  2 -3  3]
    [ 0  0  1  0  1  2  6  5]
    [ 0  0  0  1  2  1 -5  8]



Given a vector $[1,0,0,0]$, we take the value above it, in this case 0, and so on.

Now we want to calculate the reduced cost:


```python
reduced_cost = []
for i in range(M.ncols()):
    val = vector(base_values).dot_product(vector(M[2:,i])) + (-1)*M[1,i]
    reduced_cost.append(val)
reduced_cost
```




    [0, 0, 0, 0, 1, 1, 1, 0]



We compute the dot product between the vector base_values and the columns of the matrix, then we add the inverted corresponding values from the objective function to the results.


```python
M = M.rows()
M.append(reduced_cost)

matrix(M)
```




    [ a  b  c  d  x  y  z  f]
    [ 0  0  0  0 -1 -1 -1  0]
    [ 1  0  0  0  1 -1 -2  5]
    [ 0  1  0  0  1  2 -3  3]
    [ 0  0  1  0  1  2  6  5]
    [ 0  0  0  1  2  1 -5  8]
    [ 0  0  0  0  1  1  1  0]



The list created in this way is added to the initial matrix. We call this matrix a simplex matrix or simplex table.

### In the next step, we look for the central elements of the simplex table:

In the row of reduced costs (last one), we are looking for the item with the highest value. We select positive values in the column containing the largest reduced cost. Then we multiply the reciprocal of that element by the corresponding element in the last column. From the obtained results, we choose the minimum, this element is the central element.


```python
M = matrix(M)

row = list(M.row(M.nrows()-1))
max_value = max(row)
index_col = row.index(max_value)
val1 = M[2:,index_col].transpose()
val2 = M[2:,M.ncols()-1].transpose()
values = []
    
for i in range(val1.ncols()-1):
    if val1[0,i] > 0:
        values.append(val2[0,i]/val1[0,i])
    else:
        values.append(-1)
    
min_value = min([i for i in values if i > 0])
```

After that we need to get the indices of this point:


```python
index_row = values.index(min_value)
get_index = [index_row + 2, index_col]

get_index
```




    [3, 4]



### Gaussian transformation

Now we perform Gaussian transformation with respect to this point.


```python
def Gauss(A,i,j):
    A[i]=A[i]/A[i,j]
    for k in range(2,A.nrows()): #the first two lines are auxiliary, so the method starts from the third line
        if k!=i:
            A[k]=A[k]-A[k,j]*A[i]
    return A
```


```python
M1 = Gauss(M, get_index[0], get_index[1])
M1
```




    [ a  b  c  d  x  y  z  f]
    [ 0  0  0  0 -1 -1 -1  0]
    [ 1 -1  0  0  0 -3  1  2]
    [ 0  1  0  0  1  2 -3  3]
    [ 0 -1  1  0  0  0  9  2]
    [ 0 -2  0  1  0 -3  1  2]
    [ 0 -1  0  0  0 -1  4 -3]



Again, we look for the central element and perform the Gaussian transformation. We repeat the process until all the reduced costs of the matrix are non-positive.


```python
def get_point(m):
    row = list(m.row(m.nrows()-1))
    max_value = max(row)
    index_col = row.index(max_value)
    val1 = m[2:,index_col].transpose()
    val2 = m[2:,m.ncols()-1].transpose()
    values = []
    
    for i in range(val1.ncols()-1):
        if val1[0,i] > 0:
            values.append(val2[0,i]/val1[0,i])
        else:
            values.append(-1)
    
    min_value = min([i for i in values if i > 0])
    index_row = values.index(min_value)
    get_index = [index_row + 2, index_col]
    
    return get_index
```


```python
pt = get_point(M1)
M2 = Gauss(M1, pt[0], pt[1])
M2
```




    [    a     b     c     d     x     y     z     f]
    [    0     0     0     0    -1    -1    -1     0]
    [    1  -8/9  -1/9     0     0    -3     0  16/9]
    [    0   2/3   1/3     0     1     2     0  11/3]
    [    0  -1/9   1/9     0     0     0     1   2/9]
    [    0 -17/9  -1/9     1     0    -3     0  16/9]
    [    0  -5/9  -4/9     0     0    -1     0 -35/9]



### Results

All reduced costs are negative, which means that the matrix describes the so-called optimal vertex. The values of this vertex are in the last column of the matrix.


```python
values_column = vector(M2[2:M2.nrows()-1,M2.ncols()-1].transpose())
values_column
```




    (16/9, 11/3, 2/9, 16/9)



We read the vertex values as follows. If the reduced cost of a column is zero, we do the dot product of that column and the last column of the resulting matrix. In other cases, the value of the coefficient is equal to 0.


```python
txt = []

for i in range(M2.ncols()-1):
    if M2[M2.nrows()-1, i] == 0:
        txt.append(f'{M2[0,i]} = {vector(M2[2:M2.nrows()-1,i].transpose()).dot_product(values_column)}')
    else: 
        txt.append(f'{M2[0,i]} = 0')
```


```python
txt
```




    ['a = 16/9', 'b = 0', 'c = 0', 'd = 16/9', 'x = 11/3', 'y = 0', 'z = 2/9']



To read the value of the objective function, we take the reduced cost of the last column. In our case, we wanted to compute the maximum of the objective function, so we also need to change the sign of this value.


```python
txt.append(f'Max value of function is equal to:  {M2[M2.nrows()-1, M2.ncols()-1] * (-1)}')
```


```python
txt
```




    ['a = 16/9',
     'b = 0',
     'c = 0',
     'd = 16/9',
     'x = 11/3',
     'y = 0',
     'z = 2/9',
     'Max value of function is equal to:  35/9']


