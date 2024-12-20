---
title: CIFAR10 small images classification dataset
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/datasets/cifar10.py#L13" >}}

### `load_data` function

```python
keras.datasets.cifar10.load_data()
```

Loads the CIFAR10 dataset.

This is a dataset of 50,000 32x32 color training images and 10,000 test
images, labeled over 10 categories. See more info at the
[CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).

The classes are:

| Label | Description |
| ----- | ----------- |
| 0     | airplane    |
| 1     | automobile  |
| 2     | bird        |
| 3     | cat         |
| 4     | deer        |
| 5     | dog         |
| 6     | frog        |
| 7     | horse       |
| 8     | ship        |
| 9     | truck       |

**Returns**

- **Tuple of NumPy arrays**: `(x_train, y_train), (x_test, y_test)`.

**`x_train`**: `uint8` NumPy array of grayscale image data with shapes
`(50000, 32, 32, 3)`, containing the training data. Pixel values range
from 0 to 255.

**`y_train`**: `uint8` NumPy array of labels (integers in range 0-9)
with shape `(50000, 1)` for the training data.

**`x_test`**: `uint8` NumPy array of grayscale image data with shapes
`(10000, 32, 32, 3)`, containing the test data. Pixel values range
from 0 to 255.

**`y_test`**: `uint8` NumPy array of labels (integers in range 0-9)
with shape `(10000, 1)` for the test data.

**Example**

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)
```
