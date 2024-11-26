---
title: Probabilistic losses
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L497" >}}

### `BinaryCrossentropy` class

```python
keras.losses.BinaryCrossentropy(
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    reduction="sum_over_batch_size",
    name="binary_crossentropy",
    dtype=None,
)
```

Computes the cross-entropy loss between true labels and predicted labels.

Use this cross-entropy loss for binary (0 or 1) classification applications.
The loss function requires the following inputs:

- `y_true` (true label): This is either 0 or 1.
- `y_pred` (predicted value): This is the model's prediction, i.e, a single
  floating-point value which either represents a
  [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
  when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
  `from_logits=False`).

**Arguments**

- **from_logits**: Whether to interpret `y_pred` as a tensor of
  [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
  assume that `y_pred` is probabilities (i.e., values in [0, 1]).
- **label_smoothing**: Float in range [0, 1]. When 0, no smoothing occurs.
  When > 0, we compute the loss between the predicted labels
  and a smoothed version of the true labels, where the smoothing
  squeezes the labels towards 0.5. Larger values of
  `label_smoothing` correspond to heavier smoothing.
- **axis**: The axis along which to compute crossentropy (the features axis).
  Defaults to `-1`.
- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

**Examples**

**Recommended Usage:** (set `from_logits=True`)

With `compile()` API:

```python
model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    ...
)
```

As a standalone function:

```console
>>> # Example 1: (batch_size = 1, number of samples = 4)
>>> y_true = np.array([0, 1, 0, 0])
>>> y_pred = np.array([-18.6, 0.51, 2.94, -12.8])
>>> bce = keras.losses.BinaryCrossentropy(from_logits=True)
>>> bce(y_true, y_pred)
0.8654
```

```console
>>> # Example 2: (batch_size = 2, number of samples = 4)
>>> y_true = np.array([[0, 1], [0, 0]])
>>> y_pred = np.array([[-18.6, 0.51], [2.94, -12.8]])
>>> # Using default 'auto'/'sum_over_batch_size' reduction type.
>>> bce = keras.losses.BinaryCrossentropy(from_logits=True)
>>> bce(y_true, y_pred)
0.8654
>>> # Using 'sample_weight' attribute
>>> bce(y_true, y_pred, sample_weight=[0.8, 0.2])
0.243
>>> # Using 'sum' reduction` type.
>>> bce = keras.losses.BinaryCrossentropy(from_logits=True,
...     reduction="sum")
>>> bce(y_true, y_pred)
1.730
>>> # Using 'none' reduction type.
>>> bce = keras.losses.BinaryCrossentropy(from_logits=True,
...     reduction=None)
>>> bce(y_true, y_pred)
array([0.235, 1.496], dtype=float32)
```

**Default Usage:** (set `from_logits=False`)

```console
>>> # Make the following updates to the above "Recommended Usage" section
>>> # 1. Set `from_logits=False`
>>> keras.losses.BinaryCrossentropy() # OR ...('from_logits=False')
>>> # 2. Update `y_pred` to use probabilities instead of logits
>>> y_pred = [0.6, 0.3, 0.2, 0.8] # OR [[0.6, 0.3], [0.2, 0.8]]
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L616" >}}

### `BinaryFocalCrossentropy` class

```python
keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=False,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    reduction="sum_over_batch_size",
    name="binary_focal_crossentropy",
    dtype=None,
)
```

Computes focal cross-entropy loss between true labels and predictions.

Binary cross-entropy loss is often used for binary (0 or 1) classification
tasks. The loss function requires the following inputs:

- `y_true` (true label): This is either 0 or 1.
- `y_pred` (predicted value): This is the model's prediction, i.e, a single
  floating-point value which either represents a
  [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
  when `from_logits=True`) or a probability (i.e, value in `[0., 1.]` when
  `from_logits=False`).

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
helps to apply a "focal factor" to down-weight easy examples and focus more
on hard examples. By default, the focal tensor is computed as follows:

`focal_factor = (1 - output) ** gamma` for class 1
`focal_factor = output ** gamma` for class 0
where `gamma` is a focusing parameter. When `gamma=0`, this function is
equivalent to the binary crossentropy loss.

**Arguments**

- **apply_class_balancing**: A bool, whether to apply weight balancing on the
  binary classes 0 and 1.
- **alpha**: A weight balancing factor for class 1, default is `0.25` as
  mentioned in reference [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf). The weight for class 0 is
  `1.0 - alpha`.
- **gamma**: A focusing parameter used to compute the focal factor, default is
  `2.0` as mentioned in the reference
  [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).
- **from_logits**: Whether to interpret `y_pred` as a tensor of
  [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
  assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).
- **label_smoothing**: Float in `[0, 1]`. When `0`, no smoothing occurs.
  When > `0`, we compute the loss between the predicted labels
  and a smoothed version of the true labels, where the smoothing
  squeezes the labels towards `0.5`.
  Larger values of `label_smoothing` correspond to heavier smoothing.
- **axis**: The axis along which to compute crossentropy (the features axis).
  Defaults to `-1`.
- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

**Examples**

With the `compile()` API:

```python
model.compile(
    loss=keras.losses.BinaryFocalCrossentropy(
        gamma=2.0, from_logits=True),
    ...
)
```

As a standalone function:

```console
>>> # Example 1: (batch_size = 1, number of samples = 4)
>>> y_true = [0, 1, 0, 0]
>>> y_pred = [-18.6, 0.51, 2.94, -12.8]
>>> loss = keras.losses.BinaryFocalCrossentropy(
...    gamma=2, from_logits=True)
>>> loss(y_true, y_pred)
0.691
```

```console
>>> # Apply class weight
>>> loss = keras.losses.BinaryFocalCrossentropy(
...     apply_class_balancing=True, gamma=2, from_logits=True)
>>> loss(y_true, y_pred)
0.51
```

```console
>>> # Example 2: (batch_size = 2, number of samples = 4)
>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
>>> # Using default 'auto'/'sum_over_batch_size' reduction type.
>>> loss = keras.losses.BinaryFocalCrossentropy(
...     gamma=3, from_logits=True)
>>> loss(y_true, y_pred)
0.647
```

```console
>>> # Apply class weight
>>> loss = keras.losses.BinaryFocalCrossentropy(
...      apply_class_balancing=True, gamma=3, from_logits=True)
>>> loss(y_true, y_pred)
0.482
```

```console
>>> # Using 'sample_weight' attribute with focal effect
>>> loss = keras.losses.BinaryFocalCrossentropy(
...     gamma=3, from_logits=True)
>>> loss(y_true, y_pred, sample_weight=[0.8, 0.2])
0.133
```

```console
>>> # Apply class weight
>>> loss = keras.losses.BinaryFocalCrossentropy(
...      apply_class_balancing=True, gamma=3, from_logits=True)
>>> loss(y_true, y_pred, sample_weight=[0.8, 0.2])
0.097
```

```console
>>> # Using 'sum' reduction` type.
>>> loss = keras.losses.BinaryFocalCrossentropy(
...     gamma=4, from_logits=True,
...     reduction="sum")
>>> loss(y_true, y_pred)
1.222
```

```console
>>> # Apply class weight
>>> loss = keras.losses.BinaryFocalCrossentropy(
...     apply_class_balancing=True, gamma=4, from_logits=True,
...     reduction="sum")
>>> loss(y_true, y_pred)
0.914
```

```console
>>> # Using 'none' reduction type.
>>> loss = keras.losses.BinaryFocalCrossentropy(
...     gamma=5, from_logits=True,
...     reduction=None)
>>> loss(y_true, y_pred)
array([0.0017 1.1561], dtype=float32)
```

```console
>>> # Apply class weight
>>> loss = keras.losses.BinaryFocalCrossentropy(
...     apply_class_balancing=True, gamma=5, from_logits=True,
...     reduction=None)
>>> loss(y_true, y_pred)
array([0.0004 0.8670], dtype=float32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L797" >}}

### `CategoricalCrossentropy` class

```python
keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    reduction="sum_over_batch_size",
    name="categorical_crossentropy",
    dtype=None,
)
```

Computes the crossentropy loss between the labels and predictions.

Use this crossentropy loss function when there are two or more label
classes. We expect labels to be provided in a `one_hot` representation. If
you want to provide labels as integers, please use
`SparseCategoricalCrossentropy` loss. There should be `num_classes` floating
point values per feature, i.e., the shape of both `y_pred` and `y_true` are
`[batch_size, num_classes]`.

**Arguments**

- **from_logits**: Whether `y_pred` is expected to be a logits tensor. By
  default, we assume that `y_pred` encodes a probability distribution.
- **label_smoothing**: Float in [0, 1]. When > 0, label values are smoothed,
  meaning the confidence on label values are relaxed. For example, if
  `0.1`, use `0.1 / num_classes` for non-target labels and
  `0.9 + 0.1 / num_classes` for target labels.
- **axis**: The axis along which to compute crossentropy (the features
  axis). Defaults to `-1`.
- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

**Examples**

Standalone usage:

```console
>>> y_true = [[0, 1, 0], [0, 0, 1]]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> # Using 'auto'/'sum_over_batch_size' reduction type.
>>> cce = keras.losses.CategoricalCrossentropy()
>>> cce(y_true, y_pred)
1.177
```

```console
>>> # Calling with 'sample_weight'.
>>> cce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
0.814
```

```console
>>> # Using 'sum' reduction type.
>>> cce = keras.losses.CategoricalCrossentropy(
...     reduction="sum")
>>> cce(y_true, y_pred)
2.354
```

```console
>>> # Using 'none' reduction type.
>>> cce = keras.losses.CategoricalCrossentropy(
...     reduction=None)
>>> cce(y_true, y_pred)
array([0.0513, 2.303], dtype=float32)
```

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd',
              loss=keras.losses.CategoricalCrossentropy())
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L894" >}}

### `CategoricalFocalCrossentropy` class

```python
keras.losses.CategoricalFocalCrossentropy(
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    reduction="sum_over_batch_size",
    name="categorical_focal_crossentropy",
    dtype=None,
)
```

Computes the alpha balanced focal crossentropy loss.

Use this crossentropy loss function when there are two or more label
classes and if you want to handle class imbalance without using
`class_weights`. We expect labels to be provided in a `one_hot`
representation.

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
helps to apply a focal factor to down-weight easy examples and focus more on
hard examples. The general formula for the focal loss (FL)
is as follows:

`FL(p_t) = (1 - p_t) ** gamma * log(p_t)`

where `p_t` is defined as follows:
`p_t = output if y_true == 1, else 1 - output`

`(1 - p_t) ** gamma` is the `modulating_factor`, where `gamma` is a focusing
parameter. When `gamma` = 0, there is no focal effect on the cross entropy.
`gamma` reduces the importance given to simple examples in a smooth manner.

The authors use alpha-balanced variant of focal loss (FL) in the paper:
`FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t)`

where `alpha` is the weight factor for the classes. If `alpha` = 1, the
loss won't be able to handle class imbalance properly as all
classes will have the same weight. This can be a constant or a list of
constants. If alpha is a list, it must have the same length as the number
of classes.

The formula above can be generalized to:
`FL(p_t) = alpha * (1 - p_t) ** gamma * CrossEntropy(y_true, y_pred)`

where minus comes from `CrossEntropy(y_true, y_pred)` (CE).

Extending this to multi-class case is straightforward:
`FL(p_t) = alpha * (1 - p_t) ** gamma * CategoricalCE(y_true, y_pred)`

In the snippet below, there is `num_classes` floating pointing values per
example. The shape of both `y_pred` and `y_true` are
`(batch_size, num_classes)`.

**Arguments**

- **alpha**: A weight balancing factor for all classes, default is `0.25` as
  mentioned in the reference. It can be a list of floats or a scalar.
  In the multi-class case, alpha may be set by inverse class
  frequency by using `compute_class_weight` from `sklearn.utils`.
- **gamma**: A focusing parameter, default is `2.0` as mentioned in the
  reference. It helps to gradually reduce the importance given to
  simple (easy) examples in a smooth manner.
- **from_logits**: Whether `output` is expected to be a logits tensor. By
  default, we consider that `output` encodes a probability
  distribution.
- **label_smoothing**: Float in [0, 1]. When > 0, label values are smoothed,
  meaning the confidence on label values are relaxed. For example, if
  `0.1`, use `0.1 / num_classes` for non-target labels and
  `0.9 + 0.1 / num_classes` for target labels.
- **axis**: The axis along which to compute crossentropy (the features
  axis). Defaults to `-1`.
- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

**Examples**

Standalone usage:

```console
>>> y_true = [[0., 1., 0.], [0., 0., 1.]]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> # Using 'auto'/'sum_over_batch_size' reduction type.
>>> cce = keras.losses.CategoricalFocalCrossentropy()
>>> cce(y_true, y_pred)
0.23315276
```

```console
>>> # Calling with 'sample_weight'.
>>> cce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
0.1632
```

```console
>>> # Using 'sum' reduction type.
>>> cce = keras.losses.CategoricalFocalCrossentropy(
...     reduction="sum")
>>> cce(y_true, y_pred)
0.46631
```

```console
>>> # Using 'none' reduction type.
>>> cce = keras.losses.CategoricalFocalCrossentropy(
...     reduction=None)
>>> cce(y_true, y_pred)
array([3.2058331e-05, 4.6627346e-01], dtype=float32)
```

Usage with the `compile()` API:

```python
model.compile(optimizer='adam',
              loss=keras.losses.CategoricalFocalCrossentropy())
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1041" >}}

### `SparseCategoricalCrossentropy` class

```python
keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    ignore_class=None,
    reduction="sum_over_batch_size",
    name="sparse_categorical_crossentropy",
    dtype=None,
)
```

Computes the crossentropy loss between the labels and predictions.

Use this crossentropy loss function when there are two or more label
classes. We expect labels to be provided as integers. If you want to
provide labels using `one-hot` representation, please use
`CategoricalCrossentropy` loss. There should be `# classes` floating point
values per feature for `y_pred` and a single floating point value per
feature for `y_true`.

In the snippet below, there is a single floating point value per example for
`y_true` and `num_classes` floating pointing values per example for
`y_pred`. The shape of `y_true` is `[batch_size]` and the shape of `y_pred`
is `[batch_size, num_classes]`.

**Arguments**

- **from_logits**: Whether `y_pred` is expected to be a logits tensor. By
  default, we assume that `y_pred` encodes a probability distribution.
- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

**Examples**

```console
>>> y_true = [1, 2]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> # Using 'auto'/'sum_over_batch_size' reduction type.
>>> scce = keras.losses.SparseCategoricalCrossentropy()
>>> scce(y_true, y_pred)
1.177
```

```console
>>> # Calling with 'sample_weight'.
>>> scce(y_true, y_pred, sample_weight=np.array([0.3, 0.7]))
0.814
```

```console
>>> # Using 'sum' reduction type.
>>> scce = keras.losses.SparseCategoricalCrossentropy(
...     reduction="sum")
>>> scce(y_true, y_pred)
2.354
```

```console
>>> # Using 'none' reduction type.
>>> scce = keras.losses.SparseCategoricalCrossentropy(
...     reduction=None)
>>> scce(y_true, y_pred)
array([0.0513, 2.303], dtype=float32)
```

Usage with the `compile()` API:

```python
model.compile(optimizer='sgd',
              loss=keras.losses.SparseCategoricalCrossentropy())
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L466" >}}

### `Poisson` class

```python
keras.losses.Poisson(reduction="sum_over_batch_size", name="poisson", dtype=None)
```

Computes the Poisson loss between `y_true` & `y_pred`.

Formula:

```python
loss = y_pred - y_true * log(y_pred)
```

**Arguments**

- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L2024" >}}

### `CTC` class

```python
keras.losses.CTC(reduction="sum_over_batch_size", name="ctc", dtype=None)
```

CTC (Connectionist Temporal Classification) loss.

**Arguments**

- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L429" >}}

### `KLDivergence` class

```python
keras.losses.KLDivergence(
    reduction="sum_over_batch_size", name="kl_divergence", dtype=None
)
```

Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

Formula:

```python
loss = y_true * log(y_true / y_pred)
```

`y_true` and `y_pred` are expected to be probability
distributions, with values between 0 and 1. They will get
clipped to the `[0, 1]` range.

**Arguments**

- **reduction**: Type of reduction to apply to the loss. In almost all cases
  this should be `"sum_over_batch_size"`.
  Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
- **name**: Optional name for the loss instance.
- **dtype**: The dtype of the loss's computations. Defaults to `None`, which
  means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
  `"float32"` unless set to different value
  (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
  provided, then the `compute_dtype` will be utilized.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1885" >}}

### `binary_crossentropy` function

```python
keras.losses.binary_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
)
```

Computes the binary crossentropy loss.

**Arguments**

- **y_true**: Ground truth values. shape = `[batch_size, d0, .. dN]`.
- **y_pred**: The predicted values. shape = `[batch_size, d0, .. dN]`.
- **from_logits**: Whether `y_pred` is expected to be a logits tensor. By
  default, we assume that `y_pred` encodes a probability distribution.
- **label_smoothing**: Float in `[0, 1]`. If > `0` then smooth the labels by
  squeezing them towards 0.5, that is,
  using `1. - 0.5 * label_smoothing` for the target class
  and `0.5 * label_smoothing` for the non-target class.
- **axis**: The axis along which the mean is computed. Defaults to `-1`.

**Returns**

Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.

**Example**

```console
>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = keras.losses.binary_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss
array([0.916 , 0.714], dtype=float32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1663" >}}

### `categorical_crossentropy` function

```python
keras.losses.categorical_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
)
```

Computes the categorical crossentropy loss.

**Arguments**

- **y_true**: Tensor of one-hot true targets.
- **y_pred**: Tensor of predicted targets.
- **from_logits**: Whether `y_pred` is expected to be a logits tensor. By
  default, we assume that `y_pred` encodes a probability distribution.
- **label_smoothing**: Float in [0, 1]. If > `0` then smooth the labels. For
  example, if `0.1`, use `0.1 / num_classes` for non-target labels
  and `0.9 + 0.1 / num_classes` for target labels.
- **axis**: Defaults to `-1`. The dimension along which the entropy is
  computed.

**Returns**

Categorical crossentropy loss value.

**Example**

```console
>>> y_true = [[0, 1, 0], [0, 0, 1]]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> loss = keras.losses.categorical_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss
array([0.0513, 2.303], dtype=float32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1822" >}}

### `sparse_categorical_crossentropy` function

```python
keras.losses.sparse_categorical_crossentropy(
    y_true, y_pred, from_logits=False, ignore_class=None, axis=-1
)
```

Computes the sparse categorical crossentropy loss.

**Arguments**

- **y_true**: Ground truth values.
- **y_pred**: The predicted values.
- **from_logits**: Whether `y_pred` is expected to be a logits tensor. By
  default, we assume that `y_pred` encodes a probability distribution.
- **ignore_class**: Optional integer. The ID of a class to be ignored during
  loss computation. This is useful, for example, in segmentation
  problems featuring a "void" class (commonly -1 or 255) in
  segmentation maps. By default (`ignore_class=None`), all classes are
  considered.
- **axis**: Defaults to `-1`. The dimension along which the entropy is
  computed.

**Returns**

Sparse categorical crossentropy loss value.

**Examples**

```console
>>> y_true = [1, 2]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss
array([0.0513, 2.303], dtype=float32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1624" >}}

### `poisson` function

```python
keras.losses.poisson(y_true, y_pred)
```

Computes the Poisson loss between y_true and y_pred.

Formula:

```python
loss = y_pred - y_true * log(y_pred)
```

**Arguments**

- **y_true**: Ground truth values. shape = `[batch_size, d0, .. dN]`.
- **y_pred**: The predicted values. shape = `[batch_size, d0, .. dN]`.

**Returns**

Poisson loss values with shape = `[batch_size, d0, .. dN-1]`.

**Example**

```console
>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.poisson(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> y_pred = y_pred + 1e-7
>>> assert np.allclose(
...     loss, np.mean(y_pred - y_true * np.log(y_pred), axis=-1),
...     atol=1e-5)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L2055" >}}

### `ctc` function

```python
keras.losses.ctc(y_true, y_pred)
```

CTC (Connectionist Temporal Classification) loss.

**Arguments**

- **y_true**: A tensor of shape `(batch_size, max_length)` containing
  the true labels in integer format. `0` always represents
  the blank/mask index and should not be used for classes.
- **y_pred**: A tensor of shape `(batch_size, max_length, num_classes)`
  containing logits (the output of your model).
  They should _not_ be normalized via softmax.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/losses/losses.py#L1573" >}}

### `kl_divergence` function

```python
keras.losses.kl_divergence(y_true, y_pred)
```

Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

Formula:

```python
loss = y_true * log(y_true / y_pred)
```

`y_true` and `y_pred` are expected to be probability
distributions, with values between 0 and 1. They will get
clipped to the `[0, 1]` range.

**Arguments**

- **y_true**: Tensor of true targets.
- **y_pred**: Tensor of predicted targets.

**Returns**

KL Divergence loss values with shape = `[batch_size, d0, .. dN-1]`.

**Example**

```console
>>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float32)
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.kl_divergence(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> y_true = ops.clip(y_true, 1e-7, 1)
>>> y_pred = ops.clip(y_pred, 1e-7, 1)
>>> assert np.array_equal(
...     loss, np.sum(y_true * np.log(y_true / y_pred), axis=-1))
```
