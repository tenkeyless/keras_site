---
title: RandomSwap layer
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/preprocessing/random_swap.py#L18" >}}

### `RandomSwap` class

```python
keras_nlp.layers.RandomSwap(
    rate,
    max_swaps=None,
    skip_list=None,
    skip_fn=None,
    skip_py_fn=None,
    seed=None,
    name=None,
    dtype="int32",
    **kwargs
)
```

Augments input by randomly swapping words.

This layer comes in handy when you need to generate new data using swap
augmentations as described in the paper [EDA: Easy Data Augmentation
Techniques for Boosting Performance on Text Classification Tasks]
(https://arxiv.org/pdf/1901.11196.pdf). The layer expects the inputs to be
pre-split into token level inputs. This allows control over the level of
augmentation, you can split by character for character level swaps, or by
word for word level swaps.

Input data should be passed as tensors, [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor)s, or lists. For
batched input, inputs should be a list of lists or a rank two tensor. For
unbatched inputs, each element should be a list or a rank one tensor.

**Arguments**

- **rate**: The probability of a given token being chosen to be swapped
  with another random token.
- **max_swaps**: The maximum number of swaps to be performed.
- **skip_list**: A list of token values that should not be considered
  candidates for deletion.
- **skip_fn**: A function that takes as input a scalar tensor token and
  returns as output a scalar tensor True/False value. A value of
  True indicates that the token should not be considered a
  candidate for deletion. This function must be tracable–it
  should consist of tensorflow operations.
- **skip_py_fn**: A function that takes as input a python token value and
  returns as output `True` or `False`. A value of True
  indicates that should not be considered a candidate for deletion.
  Unlike the `skip_fn` argument, this argument need not be
  tracable–it can be any python function.
- **seed**: A seed for the random number generator.

**Examples**

Word level usage.

```console
>>> keras.utils.set_random_seed(1337)
>>> x = ["Hey I like", "Keras and Tensorflow"]
>>> x = list(map(lambda x: x.split(), x))
>>> augmenter = keras_hub.layers.RandomSwap(rate=0.4, seed=42)
>>> y = augmenter(x)
>>> list(map(lambda y: " ".join(y), y))
['like I Hey', 'and Keras Tensorflow']
```

Character level usage.

```console
>>> keras.utils.set_random_seed(1337)
>>> x = ["Hey Dude", "Speed Up"]
>>> x = list(map(lambda x: list(x), x))
>>> augmenter = keras_hub.layers.RandomSwap(rate=0.4, seed=42)
>>> y = augmenter(x)
>>> list(map(lambda y: "".join(y), y))
['deD yuHe', 'SUede pp']
```

Usage with skip_list.

```console
>>> keras.utils.set_random_seed(1337)
>>> x = ["Hey I like", "Keras and Tensorflow"]
>>> x = list(map(lambda x: x.split(), x))
>>> augmenter = keras_hub.layers.RandomSwap(rate=0.4,
...     skip_list=["Keras"], seed=42)
>>> y = augmenter(x)
>>> list(map(lambda y: " ".join(y), y))
['like I Hey', 'Keras and Tensorflow']
```

Usage with skip_fn.

```console
>>> def skip_fn(word):
...     return tf.strings.regex_full_match(word, r"[I, a].*")
>>> keras.utils.set_random_seed(1337)
>>> x = ["Hey I like", "Keras and Tensorflow"]
>>> x = list(map(lambda x: x.split(), x))
>>> augmenter = keras_hub.layers.RandomSwap(rate=0.9, max_swaps=3,
...     skip_fn=skip_fn, seed=11)
>>> y = augmenter(x)
>>> list(map(lambda y: " ".join(y), y))
['like I Hey', 'Keras and Tensorflow']
```

Usage with skip_py_fn.

```console
>>> def skip_py_fn(word):
...     return len(word) < 4
>>> keras.utils.set_random_seed(1337)
>>> x = ["He was drifting along", "With the wind"]
>>> x = list(map(lambda x: x.split(), x))
>>> augmenter = keras_hub.layers.RandomSwap(rate=0.8, max_swaps=2,
...     skip_py_fn=skip_py_fn, seed=15)
>>> y = augmenter(x)
>>> list(map(lambda y: " ".join(y), y))
['He was along drifting', 'wind the With']
```
