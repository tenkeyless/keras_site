---
title: RandomDeletion layer
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/layers/preprocessing/random_deletion.py#L18" >}}

### `RandomDeletion` class

```python
keras_nlp.layers.RandomDeletion(
    rate,
    max_deletions=None,
    skip_list=None,
    skip_fn=None,
    skip_py_fn=None,
    seed=None,
    name=None,
    dtype="int32",
    **kwargs
)
```

Augments input by randomly deleting tokens.

This layer comes in handy when you need to generate new data using deletion
augmentation as described in the paper [EDA: Easy Data Augmentation
Techniques for Boosting Performance on Text Classification Tasks]
(https://arxiv.org/pdf/1901.11196.pdf). The layer expects the inputs to be
pre-split into token level inputs. This allows control over the level of
augmentation, you can split by character for character level swaps, or by
word for word level swaps.

Input data should be passed as tensors, [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor)s, or lists. For
batched input, inputs should be a list of lists or a rank two tensor. For
unbatched inputs, each element should be a list or a rank one tensor.

**Arguments**

- **rate**: The probability of a token being chosen for deletion.
- **max_deletions**: The maximum number of tokens to delete.
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
>>> augmenter = keras_hub.layers.RandomDeletion(rate=0.4, seed=42)
>>> y = augmenter(x)
>>> list(map(lambda y: " ".join(y), y))
['I like', 'and']
```

Character level usage.

```console
>>> keras.utils.set_random_seed(1337)
>>> x = ["Hey Dude", "Speed Up"]
>>> x = list(map(lambda x: list(x), x))
>>> augmenter = keras_hub.layers.RandomDeletion(rate=0.4, seed=42)
>>> y = augmenter(x)
>>> list(map(lambda y: "".join(y), y))
['H Dude', 'pedUp']
```

Usage with skip_list.

```console
>>> keras.utils.set_random_seed(1337)
>>> x = ["Hey I like", "Keras and Tensorflow"]
>>> x = list(map(lambda x: x.split(), x))
>>> augmenter = keras_hub.layers.RandomDeletion(rate=0.4,
...     skip_list=["Keras", "Tensorflow"], seed=42)
>>> y = augmenter(x)
>>> list(map(lambda y: " ".join(y), y))
['I like', 'Keras Tensorflow']
```

Usage with skip_fn.

```console
>>> def skip_fn(word):
...     return tf.strings.regex_full_match(word, r"\pP")
>>> keras.utils.set_random_seed(1337)
>>> x = ["Hey I like", "Keras and Tensorflow"]
>>> x = list(map(lambda x: x.split(), x))
>>> augmenter = keras_hub.layers.RandomDeletion(rate=0.4,
...     skip_fn=skip_fn, seed=42)
>>> y = augmenter(x)
>>> list(map(lambda y: " ".join(y), y))
['I like', 'and']
```

Usage with skip_py_fn.

```console
>>> def skip_py_fn(word):
...     return len(word) < 4
>>> keras.utils.set_random_seed(1337)
>>> x = ["Hey I like", "Keras and Tensorflow"]
>>> x = list(map(lambda x: x.split(), x))
>>> augmenter = RandomDeletion(rate=0.4,
...     skip_py_fn=skip_py_fn, seed=42)
>>> y = augmenter(x)
>>> list(map(lambda y: " ".join(y), y))
['Hey I', 'and Tensorflow']
```
