---
title: Core ops
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L221" >}}

### `associative_scan` function

```python
keras.ops.associative_scan(f, elems, reverse=False, axis=0)
```

Performs a scan with an associative binary operation, in parallel.

This operation his similar to `scan`, with the key difference that
`associative_scan` is a parallel implementation with
potentially significant performance benefits, especially when jit compiled.
The catch is that it can only be used when `f` is a binary associative
operation (i.e. it must verify `f(a, f(b, c)) == f(f(a, b), c)`).

For an introduction to associative scans, refer to this paper:
Blelloch, Guy E. 1990.
[Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf).

**Arguments**

- **f**: A Python callable implementing an associative binary operation with
  signature `r = f(a, b)`. Function `f` must be associative, i.e.,
  it must satisfy the equation
  `f(a, f(b, c)) == f(f(a, b), c)`.
  The inputs and result are (possibly nested Python tree structures
  of) array(s) matching `elems`. Each array has a dimension in place
  of the `axis` dimension. `f` should be applied elementwise over
  the `axis` dimension.
  The result `r` has the same shape (and structure) as the
  two inputs `a` and `b`.
- **elems**: A (possibly nested Python tree structure of) array(s), each with
  an `axis` dimension of size `num_elems`.
- **reverse**: A boolean stating if the scan should be reversed with respect
  to the `axis` dimension.
- **axis**: an integer identifying the axis over which the scan should occur.

**Returns**

A (possibly nested Python tree structure of) array(s) of the same shape
and structure as `elems`, in which the `k`'th element of `axis` is
the result of recursively applying `f` to combine the first `k`
elements of `elems` along `axis`. For example, given
`elems = [a, b, c, ...]`, the result would be
`[a, f(a, b), f(f(a, b), c), ...]`.

**Examples**

```console
>>> sum_fn = lambda x, y: x + y
>>> xs = keras.ops.arange(5)
>>> ys = keras.ops.associative_scan(sum_fn, xs, axis=0)
>>> ys
[0, 1, 3, 6, 10]
```

```console
>>> sum_fn = lambda x, y: [x[0] + y[0], x[1] + y[1], x[2] + y[2]]
>>> xs = [keras.ops.array([[1, 2]]) for _ in range(3)]
>>> ys = keras.ops.associative_scan(sum_fn, xs, axis=0)
>>> ys
[[1, 3], [1, 3], [1, 3]]
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L782" >}}

### `cast` function

```python
keras.ops.cast(x, dtype)
```

Cast a tensor to the desired dtype.

**Arguments**

- **x**: A tensor or variable.
- **dtype**: The target type.

**Returns**

A tensor of the specified `dtype`.

**Example**

```console
>>> x = keras.ops.arange(4)
>>> x = keras.ops.cast(x, dtype="float16")
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L986" >}}

### `cond` function

```python
keras.ops.cond(pred, true_fn, false_fn)
```

Conditionally applies `true_fn` or `false_fn`.

**Arguments**

- **pred**: Boolean scalar type
- **true_fn**: Callable returning the output for the `pred == True` case.
- **false_fn**: Callable returning the output for the `pred == False` case.

**Returns**

The output of either `true_fn` or `false_fn` depending on pred.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L920" >}}

### `convert_to_numpy` function

```python
keras.ops.convert_to_numpy(x)
```

Convert a tensor to a NumPy array.

**Arguments**

- **x**: A tensor.

**Returns**

A NumPy array.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L898" >}}

### `convert_to_tensor` function

```python
keras.ops.convert_to_tensor(x, dtype=None, sparse=None)
```

Convert a NumPy array to a tensor.

**Arguments**

- **x**: A NumPy array.
- **dtype**: The target type.
- **sparse**: Whether to keep sparse tensors. `False` will cause sparse
  tensors to be densified. The default value of `None` means that
  sparse tensors are kept only if the backend supports them.

**Returns**

A tensor of the specified `dtype`.

**Example**

```console
>>> x = np.array([1, 2, 3])
>>> y = keras.ops.convert_to_tensor(x)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L1051" >}}

### `custom_gradient` function

```python
keras.ops.custom_gradient(f)
```

Decorator to define a function with a custom gradient.

This decorator allows fine grained control over the gradients of a sequence
for operations. This may be useful for multiple reasons, including providing
a more efficient or numerically stable gradient for a sequence of
operations.

**Arguments**

- **f**: Function `f(*args)` that returns a tuple
  `(output, grad_fn)`, where:
  - `args` is a sequence of (nested structures of) tensor inputs to
    the function.
  - `output` is a (nested structure of) tensor outputs of applying
    operations in `forward_fn` to `args`.
  - `grad_fn` is a function with the signature `grad_fn(*args,
upstream)` which returns a tuple of tensors the same size as
    (flattened) `args`: the derivatives of tensors in `output` with
    respect to the tensors in `args`. `upstream` is a tensor or
    sequence of tensors holding the initial value gradients for each
    tensor in `output`.

**Returns**

A function `h(*args)` which returns the same value as
`f(*args)[0]` and whose gradient is determined by
`f(*args)[1]`.

**Examples**

1. Backend-agnostic example.

```python
@ops.custom_gradient
def log1pexp(x):
    e = ops.exp(x)
    def grad(*args, upstream=None):
        if upstream is None:
            (upstream,) = args
        return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))
    return ops.log(1 + e), grad
```

Note that the grad function that returns gradient computation
requires `args` as well as an `upstream` keyword argument, depending
on the backend being set. With the JAX and TensorFlow backends,
it requires only one argument, whereas it might use the `upstream`
argument in the case of the PyTorch backend.

When working with TensorFlow/JAX backend, `grad(upstream)`
is sufficient. With PyTorch, the `grad` function requires
`*args` as well as `upstream`, e.g. `def grad(*args, upstream)`.
Follow the previous example to use `@ops.custom_gradient` in
a way that is compatible with all backends.

1. Here's JAX & TensorFlow-specific example:

```python
@ops.custom_gradient
def log1pexp(x):
    e = ops.exp(x)
    def grad(upstream):
        return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))
    return ops.log(1 + e), grad
```

1. Lastly, here's a PyTorch-specific example,
   using `*args` & `upstream`:

```python
@ops.custom_gradient
def log1pexp(x):
    e = ops.exp(x)
    def grad(*args, upstream):
        return ops.multiply(upstream, 1.0 - 1.0 / ops.add(1, e))
    return ops.log(1 + e), grad
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L746" >}}

### `dtype` function

```python
keras.ops.dtype(x)
```

Return the dtype of the tensor input as a standardized string.

Note that due to the standardization, the dtype will not compare equal
to the backend-specific version of the dtype.

**Arguments**

- **x**: A tensor. This function will try to access the `dtype` attribute of
  the input tensor.

**Returns**

A string indicating the dtype of the input tensor, e.g. `"float32"`.

**Example**

```console
>>> x = keras.ops.zeros((8, 12))
>>> keras.ops.dtype(x)
'float32'
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L898" >}}

### `erf` function

```python
keras.ops.erf(x)
```

Computes the error function of `x`, element-wise.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same dtype as `x`.

**Example**

```console
>>> x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
>>> keras.ops.erf(x)
array([-0.99998 , -0.99532, -0.842701,  0.,  0.842701], dtype=float32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L928" >}}

### `erfinv` function

```python
keras.ops.erfinv(x)
```

Computes the inverse error function of `x`, element-wise.

**Arguments**

- **x**: Input tensor.

**Returns**

A tensor with the same dtype as `x`.

**Example**

```console
>>> x = np.array([-0.5, -0.2, -0.1, 0.0, 0.3])
>>> keras.ops.erfinv(x)
array([-0.47694, -0.17914, -0.08886,  0. ,  0.27246], dtype=float32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L297" >}}

### `extract_sequences` function

```python
keras.ops.extract_sequences(x, sequence_length, sequence_stride)
```

Expands the dimension of last axis into sequences of `sequence_length`.

Slides a window of size `sequence_length` over the last axis of the input
with a stride of `sequence_stride`, replacing the last axis with
`[num_sequences, sequence_length]` sequences.

If the dimension along the last axis is N, the number of sequences can be
computed by:

`num_sequences = 1 + (N - sequence_length) // sequence_stride`

**Arguments**

- **x**: Input tensor.
- **sequence_length**: An integer representing the sequences length.
- **sequence_stride**: An integer representing the sequences hop size.

**Returns**

A tensor of sequences with shape [..., num\_sequences, sequence\_length].

**Example**

```console
>>> x = keras.ops.convert_to_tensor([1, 2, 3, 4, 5, 6])
>>> extract_sequences(x, 3, 2)
array([[1, 2, 3],
   [3, 4, 5]])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L636" >}}

### `fori_loop` function

```python
keras.ops.fori_loop(lower, upper, body_fun, init_val)
```

For loop implementation.

**Arguments**

- **lower**: The initial value of the loop variable.
- **upper**: The upper bound of the loop variable.
- **body_fun**: A callable that represents the loop body. Must take two
  arguments: the loop variable and the loop state. The loop state
  should be updated and returned by this function.
- **init_val**: The initial value of the loop state.

**Returns**

The final state after the loop.

**Example**

```console
>>> lower = 0
>>> upper = 10
>>> body_fun = lambda i, s: (i + 1, s + i)
>>> init_val = 0
>>> keras.ops.fori_loop(lower, upper, body_fun, init_val)
45
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L199" >}}

### `in_top_k` function

```python
keras.ops.in_top_k(targets, predictions, k)
```

Checks if the targets are in the top-k predictions.

**Arguments**

- **targets**: A tensor of true labels.
- **predictions**: A tensor of predicted labels.
- **k**: An integer representing the number of predictions to consider.

**Returns**

A boolean tensor of the same shape as `targets`, where each element
indicates whether the corresponding target is in the top-k predictions.

**Example**

```console
>>> targets = keras.ops.convert_to_tensor([2, 5, 3])
>>> predictions = keras.ops.convert_to_tensor(
... [[0.1, 0.4, 0.6, 0.9, 0.5],
...  [0.1, 0.7, 0.9, 0.8, 0.3],
...  [0.1, 0.6, 0.9, 0.9, 0.5]])
>>> in_top_k(targets, predictions, k=3)
array([ True False  True], shape=(3,), dtype=bool)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L1035" >}}

### `is_tensor` function

```python
keras.ops.is_tensor(x)
```

Check whether the given object is a tensor.

Note: This checks for backend specific tensors so passing a TensorFlow
tensor would return `False` if your backend is PyTorch or JAX.

**Arguments**

- **x**: A variable.

**Returns**

`True` if `x` is a tensor, otherwise `False`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L241" >}}

### `logsumexp` function

```python
keras.ops.logsumexp(x, axis=None, keepdims=False)
```

Computes the logarithm of sum of exponentials of elements in a tensor.

**Arguments**

- **x**: Input tensor.
- **axis**: An integer or a tuple of integers specifying the axis/axes
  along which to compute the sum. If `None`, the sum is computed
  over all elements. Defaults to `None`.
- **keepdims**: A boolean indicating whether to keep the dimensions of
  the input tensor when computing the sum. Defaults to `False`.

**Returns**

A tensor containing the logarithm of the sum of exponentials of
elements in `x`.

**Example**

```console
>>> x = keras.ops.convert_to_tensor([1., 2., 3.])
>>> logsumexp(x)
3.407606
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L34" >}}

### `map` function

```python
keras.ops.map(f, xs)
```

Map a function over leading array axes.

Like Pythonâs builtin map, except inputs and outputs are in the form of
stacked arrays. Consider using the `vectorized_map()` transform instead,
unless you need to apply a function element by element for reduced memory
usage or heterogeneous computation with other control flow primitives.

When `xs` is an array type, the semantics of `map()` are given by this
Python implementation:

```python
def map(f, xs):
    return np.stack([f(x) for x in xs])
```

**Arguments**

- **f**: Callable defines the function to apply element-wise over the first
  axis or axes of `xs`.
- **xs**: Values over which to map along the leading axis.

**Returns**

Mapped values.

**Examples**

```console
>>> f = lambda x: x**2
>>> xs = keras.ops.arange(10)
>>> ys = keras.ops.map(f, xs)
>>> ys
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

```console
>>> f = lambda x: {"y1": x**2, "y2": x * 10}  # Can have nested outputs
>>> ys = keras.ops.map(f, xs)
>>> ys["y1"]
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
>>> ys["y2"]
[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L868" >}}

### `rsqrt` function

```python
keras.ops.rsqrt(x)
```

Computes reciprocal of square root of x element-wise.

**Arguments**

- **x**: input tensor

**Returns**

A tensor with the same dtype as `x`.

**Example**

```console
>>> x = keras.ops.convert_to_tensor([1.0, 10.0, 100.0])
>>> keras.ops.rsqrt(x)
array([1.0, 0.31622776, 0.1], dtype=float32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L817" >}}

### `saturate_cast` function

```python
keras.ops.saturate_cast(x, dtype)
```

Performs a safe saturating cast to the desired dtype.

Saturating cast prevents data type overflow when casting to `dtype` with
smaller values range. E.g.
`ops.cast(ops.cast([-1, 256], "float32"), "uint8")` returns `[255, 0]`,
but `ops.saturate_cast(ops.cast([-1, 256], "float32"), "uint8")` returns
`[0, 255]`.

**Arguments**

- **x**: A tensor or variable.
- **dtype**: The target type.

**Returns**

A safely casted tensor of the specified `dtype`.

**Example**

Image resizing with bicubic interpolation may produce values outside
original range.

```console
>>> image2x2 = np.array([0, 1, 254, 255], dtype="uint8").reshape(1, 2, 2, 1)
>>> image4x4 = tf.image.resize(image2x2, (4, 4), method="bicubic")
>>> print(image4x4.numpy().squeeze())
>>> # [[-22.500004 -22.204624 -21.618908 -21.32353 ]
>>> #  [ 52.526054  52.82143   53.407146  53.70253 ]
>>> #  [201.29752  201.59288  202.17859  202.47395 ]
>>> #  [276.32355  276.61893  277.20465  277.50006 ]]
```

Casting this resized image back to `uint8` will cause overflow.

```console
>>> image4x4_casted = ops.cast(image4x4, "uint8")
>>> print(image4x4_casted.numpy().squeeze())
>>> # [[234 234 235 235]
>>> #  [ 52  52  53  53]
>>> #  [201 201 202 202]
>>> #  [ 20  20  21  21]]
```

Saturate casting to `uint8` will clip values to `uint8` range before
casting and will not cause overflow.

```console
>>> image4x4_saturate_casted = ops.saturate_cast(image4x4, "uint8")
>>> print(image4x4_saturate_casted.numpy().squeeze())
>>> # [[  0   0   0   0]
>>> #  [ 52  52  53  53]
>>> #  [201 201 202 202]
>>> #  [255 255 255 255]]
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L107" >}}

### `scan` function

```python
keras.ops.scan(f, init, xs=None, length=None, reverse=False, unroll=1)
```

Scan a function over leading array axes while carrying along state.

When the type of `xs` is an array type or `None`, and the type of `ys` is an
array type, the semantics of `scan()` are given roughly by this Python
implementation:

```python
def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)
```

The loop-carried value `carry` (`init`) must hold a fixed shape and dtype
across all iterations.

In TensorFlow, `y` must match `carry` in shape and dtype. This is not
required in other backends.

**Arguments**

- **f**: Callable defines the logic for each loop iteration. This accepts two
  arguments where the first is a value of the loop carry and the
  second is a slice of `xs` along its leading axis.
  This callable returns a pair where the first represents a new value
  for the loop carry and the second represents a slice of the output.
- **init**: The initial loop carry value. This can be a scalar, tensor, or any
  nested structure. It must match the structure of the first element
  returned by `f`.
- **xs**: Optional value to scan along its leading axis. This can be a tensor
  or any nested structure. If `xs` is not provided, you must specify
  `length` to define the number of loop iterations.
  Defaults to `None`.
- **length**: Optional integer specifying the number of loop iterations.
  If `length` is not provided, it defaults to the sizes of leading
  axis of the arrays in `xs`. Defaults to `None`.
- **reverse**: Optional boolean specifying whether to run the scan iteration
  forward or in reverse, equivalent to reversing the leading axes of
  the arrays in both `xs` and in `ys`.
- **unroll**: Optional positive integer or boolean specifying how many scan
  iterations to unroll within a single iteration of a loop. If an
  integer is provided, it determines how many unrolled loop iterations
  to run within a single rolled iteration of the loop. If a boolean is
  provided, it will determine if the loop is completely unrolled
  (`unroll=True`) or left completely unrolled (`unroll=False`).
  Note that unrolling is only supported by JAX and TensorFlow
  backends.

**Returns**

A pair where the first element represents the final loop carry value and
the second element represents the stacked outputs of `f` when scanned
over the leading axis of the inputs.

**Examples**

```console
>>> sum_fn = lambda c, x: (c + x, c + x)
>>> init = keras.ops.array(0)
>>> xs = keras.ops.array([1, 2, 3, 4, 5])
>>> carry, result = keras.ops.scan(sum_fn, init, xs)
>>> carry
15
>>> result
[1, 3, 6, 10, 15]
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L288" >}}

### `scatter` function

```python
keras.ops.scatter(indices, values, shape)
```

Returns a tensor of shape `shape` where `indices` are set to `values`.

At a high level, this operation does `zeros[indices] = updates` and
returns the output. It is equivalent to:

```python
zeros = keras.ops.zeros(shape)
output = keras.ops.scatter_update(zeros, indices, values)
```

**Arguments**

- **indices**: A tensor or list/tuple specifying
  indices for the values in `values`.
- **values**: A tensor, the values to be set at `indices`.
- **shape**: Shape of the output tensor.

**Example**

```console
>>> indices = [[0, 1], [1, 1]]
>>> values = np.array([1., 1.])
>>> keras.ops.scatter(indices, values, shape=(2, 2))
array([[0., 1.],
       [0., 1.]])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L327" >}}

### `scatter_update` function

```python
keras.ops.scatter_update(inputs, indices, updates)
```

Update inputs via updates at scattered (sparse) indices.

At a high level, this operation does `inputs[indices] = updates`.
Assume `inputs` is a tensor of shape `(D0, D1, ..., Dn)`, there are 2 main
usages of `scatter_update`.

1. `indices` is a 2D tensor of shape `(num_updates, n)`, where `num_updates`
   is the number of updates to perform, and `updates` is a 1D tensor of
   shape `(num_updates,)`. For example, if `inputs` is `zeros((4, 4, 4))`,
   and we want to update `inputs[1, 2, 3]` and `inputs[0, 1, 3]` as 1, then
   we can use:

```python
inputs = np.zeros((4, 4, 4))
indices = [[1, 2, 3], [0, 1, 3]]
updates = np.array([1., 1.])
inputs = keras.ops.scatter_update(inputs, indices, updates)
```

2 `indices` is a 2D tensor of shape `(num_updates, k)`, where `num_updates`
is the number of updates to perform, and `k` (`k < n`) is the size of
each index in `indices`. `updates` is a `n - k`-D tensor of shape
`(num_updates, inputs.shape[k:])`. For example, if
`inputs = np.zeros((4, 4, 4))`, and we want to update `inputs[1, 2, :]`
and `inputs[2, 3, :]` as `[1, 1, 1, 1]`, then `indices` would have shape
`(num_updates, 2)` (`k = 2`), and `updates` would have shape
`(num_updates, 4)` (`inputs.shape[2:] = 4`). See the code below:

```python
inputs = np.zeros((4, 4, 4))
indices = [[1, 2], [2, 3]]
updates = np.array([[1., 1., 1, 1,], [1., 1., 1, 1,])
inputs = keras.ops.scatter_update(inputs, indices, updates)
```

**Arguments**

- **inputs**: A tensor, the tensor to be updated.
- **indices**: A tensor or list/tuple of shape `(N, inputs.ndim)`, specifying
  indices to update. `N` is the number of indices to update, must be
  equal to the first dimension of `updates`.
- **updates**: A tensor, the new values to be put to `inputs` at `indices`.

**Returns**

A tensor, has the same shape and dtype as `inputs`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L104" >}}

### `segment_max` function

```python
keras.ops.segment_max(data, segment_ids, num_segments=None, sorted=False)
```

Computes the max of segments in a tensor.

**Arguments**

- **data**: Input tensor.
- **segment_ids**: A N-D tensor containing segment indices for each
  element in `data`. data.shape[:len(segment\_ids.shape)] should match.
- **num_segments**: An integer representing the total number of
  segments. If not specified, it is inferred from the maximum
  value in `segment_ids`.
- **sorted**: A boolean indicating whether `segment_ids` is sorted.
  Defaults to `False`.

**Returns**

A tensor containing the max of segments, where each element
represents the max of the corresponding segment in `data`.

**Example**

```console
>>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])
>>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
>>> num_segments = 3
>>> keras.ops.segment_max(data, segment_ids, num_segments)
array([2, 20, 200], dtype=int32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L57" >}}

### `segment_sum` function

```python
keras.ops.segment_sum(data, segment_ids, num_segments=None, sorted=False)
```

Computes the sum of segments in a tensor.

**Arguments**

- **data**: Input tensor.
- **segment_ids**: A N-D tensor containing segment indices for each
  element in `data`. Num dims for segment ids should be strictly
  smaller or equal to number of dims in data.
- **num_segments**: An integer representing the total number of
  segments. If not specified, it is inferred from the maximum
  value in `segment_ids`.
- **sorted**: A boolean indicating whether `segment_ids` is sorted.
  Defaults to `False`.

**Returns**

A tensor containing the sum of segments, where each element
represents the sum of the corresponding segment in `data`.

**Example**

```console
>>> data = keras.ops.convert_to_tensor([1, 2, 10, 20, 100, 200])
>>> segment_ids = keras.ops.convert_to_tensor([0, 0, 1, 1, 2, 2])
>>> num_segments = 3
>>> keras.ops.segment_sum(data, segment_ids,num_segments)
array([3, 30, 300], dtype=int32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L719" >}}

### `shape` function

```python
keras.ops.shape(x)
```

Gets the shape of the tensor input.

Note: On the TensorFlow backend, when `x` is a [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) with dynamic
shape, dimensions which are dynamic in the context of a compiled function
will have a [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) value instead of a static integer value.

**Arguments**

- **x**: A tensor. This function will try to access the `shape` attribute of
  the input tensor.

**Returns**

A tuple of integers or None values, indicating the shape of the input
tensor.

**Example**

```console
>>> x = keras.ops.zeros((8, 12))
>>> keras.ops.shape(x)
(8, 12)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L387" >}}

### `slice` function

```python
keras.ops.slice(inputs, start_indices, shape)
```

Return a slice of an input tensor.

At a high level, this operation is an explicit replacement for array slicing
e.g. `inputs[start_indices: start_indices + shape]`.
Unlike slicing via brackets, this operation will accept tensor start
indices on all backends, which is useful when indices dynamically computed
via other tensor operations.

```python
inputs = np.zeros((5, 5))
start_indices = np.array([3, 3])
shape = np.array([2, 2])
inputs = keras.ops.slice(inputs, start_indices, shape)
```

**Arguments**

- **inputs**: A tensor, the tensor to be updated.
- **start_indices**: A list/tuple of shape `(inputs.ndim,)`, specifying
  the starting indices for updating.
- **shape**: The full shape of the returned slice.

**Returns**

A tensor, has the same shape and dtype as `inputs`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L426" >}}

### `slice_update` function

```python
keras.ops.slice_update(inputs, start_indices, updates)
```

Update an input by slicing in a tensor of updated values.

At a high level, this operation does
`inputs[start_indices: start_indices + updates.shape] = updates`.
Assume inputs is a tensor of shape `(D0, D1, ..., Dn)`,
`start_indices` must be a list/tuple of n integers, specifying the starting
indices. `updates` must have the same rank as `inputs`, and the size of each
dim must not exceed `Di - start_indices[i]`. For example, if we have 2D
inputs `inputs = np.zeros((5, 5))`, and we want to update the intersection
of last 2 rows and last 2 columns as 1, i.e.,
`inputs[3:, 3:] = np.ones((2, 2))`, then we can use the code below:

```python
inputs = np.zeros((5, 5))
start_indices = [3, 3]
updates = np.ones((2, 2))
inputs = keras.ops.slice_update(inputs, start_indices, updates)
```

**Arguments**

- **inputs**: A tensor, the tensor to be updated.
- **start_indices**: A list/tuple of shape `(inputs.ndim,)`, specifying
  the starting indices for updating.
- **updates**: A tensor, the new values to be put to `inputs` at `indices`.
  `updates` must have the same rank as `inputs`.

**Returns**

A tensor, has the same shape and dtype as `inputs`.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L593" >}}

### `stop_gradient` function

```python
keras.ops.stop_gradient(variable)
```

Stops gradient computation.

**Arguments**

- **variable**: A tensor variable for which the gradient
  computation is to be disabled.

**Returns**

The variable with gradient computation disabled.

**Examples**

```console
>>> var = keras.backend.convert_to_tensor(
...     [1., 2., 3.],
...     dtype="float32"
... )
>>> var = keras.ops.stop_gradient(var)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L472" >}}

### `switch` function

```python
keras.ops.switch(index, branches, *operands)
```

Apply exactly one of the `branches` given by `index`.

If `index` is out of bounds, it is clamped to within bounds.

The semantics of `switch` are given roughly by this Python implementation:

```python
def switch(index, branches, *operands):
    index = clamp(0, index, len(branches) - 1)
    return branches[index](*operands)
```

**Arguments**

- **index**: An integer scalar indicating which branch function to apply.
- **branches**: A sequence of functions to be applied based on `index`.
- **operands**: Inputs to whichever branch is applied.

**Returns**

The outputs of `branch(*operands)` for the branch that was selected
based on `index`.

**Examples**

```console
>>> add_fn = lambda x, y: x + y
>>> subtract_fn = lambda x, y: x - y
>>> x = keras.ops.array(2.0)
>>> y = keras.ops.array(0.5)
>>> branches = [add_fn, subtract_fn]
>>> keras.ops.switch(0, branches, x, y)
2.5
```

```console
>>> keras.ops.switch(1, branches, x, y)
1.5
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L157" >}}

### `top_k` function

```python
keras.ops.top_k(x, k, sorted=True)
```

Finds the top-k values and their indices in a tensor.

**Arguments**

- **x**: Input tensor.
- **k**: An integer representing the number of top elements to retrieve.
- **sorted**: A boolean indicating whether to sort the output in
  descending order. Defaults to `True`.

**Returns**

A tuple containing two tensors. The first tensor contains the
top-k values, and the second tensor contains the indices of the
top-k values in the input tensor.

**Example**

```console
>>> x = keras.ops.convert_to_tensor([5, 2, 7, 1, 9, 3])
>>> values, indices = top_k(x, k=3)
>>> print(values)
array([9 7 5], shape=(3,), dtype=int32)
>>> print(indices)
array([4 2 0], shape=(3,), dtype=int32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L695" >}}

### `unstack` function

```python
keras.ops.unstack(x, num=None, axis=0)
```

Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.

**Arguments**

- **x**: The input tensor.
- **num**: The length of the dimension axis. Automatically inferred
  if `None`.
- **axis**: The axis along which to unpack.

**Returns**

A list of tensors unpacked along the given axis.

**Example**

```console
>>> x = keras.ops.array([[1, 2], [3, 4]])
>>> keras.ops.unstack(x, axis=0)
[array([1, 2]), array([3, 4])]
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L1002" >}}

### `vectorized_map` function

```python
keras.ops.vectorized_map(function, elements)
```

Parallel map of `function` on axis 0 of tensor(s) `elements`.

Schematically, `vectorized_map` implements the following,
in the case of a single tensor input `elements`:

```python
def vectorized_map(function, elements)
    outputs = []
    for e in elements:
        outputs.append(function(e))
    return stack(outputs)
```

In the case of an iterable of tensors `elements`,
it implements the following:

```python
def vectorized_map(function, elements)
    batch_size = elements[0].shape[0]
    outputs = []
    for index in range(batch_size):
        outputs.append(function([e[index] for e in elements]))
    return np.stack(outputs)
```

In this case, `function` is expected to take as input
a single list of tensor arguments.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/core.py#L532" >}}

### `while_loop` function

```python
keras.ops.while_loop(cond, body, loop_vars, maximum_iterations=None)
```

While loop implementation.

**Arguments**

- **cond**: A callable that represents the termination condition of the loop.
  Must accept a `loop_vars` like structure as an argument. If
  `loop_vars` is a tuple or list, each element of `loop_vars` will be
  passed positionally to the callable.
- **body**: A callable that represents the loop body. Must accept a
  `loop_vars` like structure as an argument, and return update value
  with the same structure. If `loop_vars` is a tuple or list, each
  element of `loop_vars` will be passed positionally to the callable.
- **loop_vars**: An arbitrary nested structure of tensor state to persist
  across loop iterations.
- **maximum_iterations**: Optional maximum number of iterations of the while
  loop to run. If provided, the `cond` output is AND-ed with an
  additional condition ensuring the number of iterations executed is
  no greater than `maximum_iterations`.

**Returns**

A list/tuple of tensors, has the same shape and dtype as `inputs`.

**Examples**

```console
>>> i = 0
>>> cond = lambda i: i < 10
>>> body = lambda i: i + 1
>>> keras.ops.while_loop(cond, body, i)
10
```

```console
>>> x, y = 0, 1
>>> cond = lambda x, y: x < 10
>>> body = lambda x, y: (x + 1, y + 1)
>>> keras.ops.while_loop(cond, body, (x, y))
10, 11
```
