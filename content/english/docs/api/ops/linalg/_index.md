---
title: Linear algebra ops
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L22" >}}

### `cholesky` function

```python
keras.ops.cholesky(x)
```

Computes the Cholesky decomposition of a positive semi-definite matrix.

**Arguments**

- **x**: Input tensor of shape `(..., M, M)`.

**Returns**

A tensor of shape `(..., M, M)` representing the lower triangular
Cholesky factor of `x`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L63" >}}

### `det` function

```python
keras.ops.det(x)
```

Computes the determinant of a square tensor.

**Arguments**

- **x**: Input tensor of shape `(..., M, M)`.

**Returns**

A tensor of shape `(...,)` representing the determinant of `x`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L103" >}}

### `eig` function

```python
keras.ops.eig(x)
```

Computes the eigenvalues and eigenvectors of a square matrix.

**Arguments**

- **x**: Input tensor of shape `(..., M, M)`.

**Returns**

- **A tuple of two tensors**: a tensor of shape `(..., M)` containing
  eigenvalues and a tensor of shape `(..., M, M)` containing eigenvectors.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L143" >}}

### `eigh` function

```python
keras.ops.eigh(x)
```

Computes the eigenvalues and eigenvectors of a complex Hermitian.

**Arguments**

- **x**: Input tensor of shape `(..., M, M)`.

**Returns**

- **A tuple of two tensors**: a tensor of shape `(..., M)` containing
  eigenvalues and a tensor of shape `(..., M, M)` containing eigenvectors.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L181" >}}

### `inv` function

```python
keras.ops.inv(x)
```

Computes the inverse of a square tensor.

**Arguments**

- **x**: Input tensor of shape `(..., M, M)`.

**Returns**

A tensor of shape `(..., M, M)` representing the inverse of `x`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L961" >}}

### `logdet` function

```python
keras.ops.logdet(x)
```

Computes log of the determinant of a hermitian positive definite matrix.

**Arguments**

- **x**: Input matrix. It must 2D and square.

**Returns**

The natural log of the determinant of matrix.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L629" >}}

### `lstsq` function

```python
keras.ops.lstsq(a, b, rcond=None)
```

Return the least-squares solution to a linear matrix equation.

Computes the vector x that approximately solves the equation
`a @ x = b`. The equation may be under-, well-, or over-determined
(i.e., the number of linearly independent rows of a can be less than,
equal to, or greater than its number of linearly independent columns).
If a is square and of full rank, then `x` (but for round-off error)
is the exact solution of the equation. Else, `x` minimizes the
L2 norm of `b - a * x`.

If there are multiple minimizing solutions,
the one with the smallest L2 norm is returned.

**Arguments**

- **a**: "Coefficient" matrix of shape `(M, N)`.
- **b**: Ordinate or "dependent variable" values,
  of shape `(M,)` or `(M, K)`.
  If `b` is two-dimensional, the least-squares solution
  is calculated for each of the K columns of `b`.
- **rcond**: Cut-off ratio for small singular values of `a`.
  For the purposes of rank determination,
  singular values are treated as zero if they are
  smaller than rcond times the largest
  singular value of `a`.

**Returns**

Tensor with shape `(N,)` or `(N, K)` containing
the least-squares solutions.

**NOTE:** The output differs from `numpy.linalg.lstsq`.
NumPy returns a tuple with four elements, the first of which
being the least-squares solutions and the others
being essentially never used.
Keras only returns the first value. This is done both
to ensure consistency across backends (which cannot be achieved
for the other values) and to simplify the API.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L223" >}}

### `lu_factor` function

```python
keras.ops.lu_factor(x)
```

Computes the lower-upper decomposition of a square matrix.

**Arguments**

- **x**: A tensor of shape `(..., M, M)`.

**Returns**

- **A tuple of two tensors**: a tensor of shape `(..., M, M)` containing the
  lower and upper triangular matrices and a tensor of shape `(..., M)`
  containing the pivots.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L312" >}}

### `norm` function

```python
keras.ops.norm(x, ord=None, axis=None, keepdims=False)
```

Matrix or vector norm.

This function is able to return one of eight different matrix norms, or one
of an infinite number of vector norms (described below), depending on the
value of the `ord` parameter.

**Arguments**

- **x**: Input tensor.
- **ord**: Order of the norm (see table under Notes). The default is `None`.
- **axis**: If `axis` is an integer, it specifies the axis of `x` along which
  to compute the vector norms. If `axis` is a 2-tuple, it specifies
  the axes that hold 2-D matrices, and the matrix norms of these
  matrices are computed.
- **keepdims**: If this is set to `True`, the axes which are reduced are left
  in the result as dimensions with size one.

Note:
For values of `ord < 1`, the result is, strictly speaking, not a
mathematical 'norm', but it may still be useful for various numerical
purposes. The following norms can be calculated:

- For matrices:
- `ord=None`: Frobenius norm
- `ord="fro"`: Frobenius norm
- `ord="nuc"`: nuclear norm
- `ord=np.inf`: `max(sum(abs(x), axis=1))`
- `ord=-np.inf`: `min(sum(abs(x), axis=1))`
- `ord=0`: not supported
- `ord=1`: `max(sum(abs(x), axis=0))`
- `ord=-1`: `min(sum(abs(x), axis=0))`
- `ord=2`: 2-norm (largest sing. value)
- `ord=-2`: smallest singular value
- other: not supported
- For vectors:
- `ord=None`: 2-norm
- `ord="fro"`: not supported
- `ord="nuc"`: not supported
- `ord=np.inf`: `max(abs(x))`
- `ord=-np.inf`: `min(abs(x))`
- `ord=0`: `sum(x != 0)`
- `ord=1`: as below
- `ord=-1`: as below
- `ord=2`: as below
- `ord=-2`: as below
- other: `sum(abs(x)**ord)**(1./ord)`

**Returns**

Norm of the matrix or vector(s).

**Example**

```console
>>> x = keras.ops.reshape(keras.ops.arange(9, dtype="float32") - 4, (3, 3))
>>> keras.ops.linalg.norm(x)
7.7459664
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L417" >}}

### `qr` function

```python
keras.ops.qr(x, mode="reduced")
```

Computes the QR decomposition of a tensor.

**Arguments**

- **x**: Input tensor of shape `(..., M, N)`.
- **mode**: A string specifying the mode of the QR decomposition.
  - 'reduced': Returns the reduced QR decomposition. (default)
  - 'complete': Returns the complete QR decomposition.

**Returns**

A tuple containing two tensors. The first tensor of shape `(..., M, K)`
is the orthogonal matrix `q` and the second tensor of shape
`(..., K, N)` is the upper triangular matrix `r`, where `K = min(M, N)`.

**Example**

```console
>>> x = keras.ops.convert_to_tensor([[1., 2.], [3., 4.], [5., 6.]])
>>> q, r = qr(x)
>>> print(q)
array([[-0.16903079  0.897085]
       [-0.5070925   0.2760267 ]
       [-0.8451542  -0.34503305]], shape=(3, 2), dtype=float32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L463" >}}

### `solve` function

```python
keras.ops.solve(a, b)
```

Solves a linear system of equations given by `a x = b`.

**Arguments**

- **a**: A tensor of shape `(..., M, M)` representing the coefficients matrix.
- **b**: A tensor of shape `(..., M)` or `(..., M, N)` represeting the
  right-hand side or "dependent variable" matrix.

**Returns**

A tensor of shape `(..., M)` or `(..., M, N)` representing the solution
of the linear system. Returned shape is identical to `b`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L509" >}}

### `solve_triangular` function

```python
keras.ops.solve_triangular(a, b, lower=False)
```

Solves a linear system of equations given by `a x = b`.

**Arguments**

- **a**: A tensor of shape `(..., M, M)` representing the coefficients matrix.
- **b**: A tensor of shape `(..., M)` or `(..., M, N)` represeting the
  right-hand side or "dependent variable" matrix.

**Returns**

A tensor of shape `(..., M)` or `(..., M, N)` representing the solution
of the linear system. Returned shape is identical to `b`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/linalg.py#L571" >}}

### `svd` function

```python
keras.ops.svd(x, full_matrices=True, compute_uv=True)
```

Computes the singular value decomposition of a matrix.

**Arguments**

- **x**: Input tensor of shape `(..., M, N)`.

**Returns**

- **A tuple of three tensors**: a tensor of shape `(..., M, M)` containing the
  left singular vectors, a tensor of shape `(..., M, N)` containing the
  singular values and a tensor of shape `(..., N, N)` containing the
  right singular vectors.
