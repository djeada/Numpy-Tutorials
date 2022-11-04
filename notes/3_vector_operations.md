
<h1>About vectors</h1>

A vector in R^n is an n-tuple. 

In a <b>row vector</b>, the elements of the vector are written next to each other, and in a <b>column vector</b>, the elements of the vector are written on top of each other.

A column <b>vector's transpose</b> is a row vector of the same length, and a row vector's transpose is a column vector.

The <b>norm</b> is a way to measure vector's length. Depending on the metric used, there are a variety of ways to define the length of a vector . The most common is L2 norm, which uses the distance formula.

$$ ||\vec{v}||_p = \sqrt[p]{\sum_i v_i^p} $$
 
Operations:

1) Vector addition: 
The pairwise addition of respective elements.

```Python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.add(arr_1, arr_2))
```

Expected output:

```
[ 6 10  7]
```

2) Scalar multiplication:
The product of each element of the vector by the given scalar.

```Python
arr = np.array([6, 3, 4])
scalar = 2
print(scalar * arr)
```

Expected output:

```
[12  6  8]
```

3) Dot product:
The sum of pairwise products of respective elements.

$$\vec{v} \cdot \vec{w}= |\vec{v}| |\vec{w}| \cos\theta $$

```Python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.dot(arr_1, arr_2))
```

Expected output:

```
-1
```

4) The cross product:

  The cross product's geometric representation is a vector perpendicular to both v and w, with a length equal to the region enclosed by the parallelogram formed by the two vectors.
 
$$\vec{v}\times\vec{w} =|\vec{v}||\vec{w}|\sin\theta$$

  Where <i>n</i> is a unit vector perpendicular to plane.

```Python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.cross(arr_1, arr_2))
```

Expected output:

```
[[-36 -33  78]]
```

5) The angle between two vectors:

$$\theta = \arcsin{\frac{\vec{v}\times\vec{w}}{|\vec{v}||\vec{w}|}}$$

If the angle between the vectors θ=π/2, then the vectors are said to be perpendicular or orthogonal, and the dot product is 0.

```Python
arr_1 = np.array([9, 2, 5])
arr_2 = np.array([-3, 8, 2])
print(np.arccos(np.dot(arr_1, arr_2)/(np.norm(arr_1)*np.norm(arr_2)))
```

Expected output:

```
1.582
```
