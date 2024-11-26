---
title: IMDB movie review sentiment classification dataset
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/datasets/imdb.py#L12" >}}

### `load_data` function

```python
keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
    **kwargs
)
```

Loads the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment
(positive/negative). Reviews have been preprocessed, and each review is
encoded as a list of word indexes (integers).
For convenience, words are indexed by overall frequency in the dataset,
so that for instance the integer "3" encodes the 3rd most frequent word in
the data. This allows for quick filtering operations such as:
"only consider the top 10,000 most
common words, but eliminate the top 20 most common words".

As a convention, "0" does not stand for a specific word, but instead is used
to encode the pad token.

**Arguments**

- **path**: where to cache the data (relative to `~/.keras/dataset`).
- **num_words**: integer or None. Words are
  ranked by how often they occur (in the training set) and only
  the `num_words` most frequent words are kept. Any less frequent word
  will appear as `oov_char` value in the sequence data. If None,
  all words are kept. Defaults to `None`.
- **skip_top**: skip the top N most frequently occurring words
  (which may not be informative). These words will appear as
  `oov_char` value in the dataset. When 0, no words are
  skipped. Defaults to `0`.
- **maxlen**: int or None. Maximum sequence length.
  Any longer sequence will be truncated. None, means no truncation.
  Defaults to `None`.
- **seed**: int. Seed for reproducible data shuffling.
- **start_char**: int. The start of a sequence will be marked with this
  character. 0 is usually the padding character. Defaults to `1`.
- **oov_char**: int. The out-of-vocabulary character.
  Words that were cut out because of the `num_words` or
  `skip_top` limits will be replaced with this character.
- **index_from**: int. Index actual words with this index and higher.

**Returns**

- **Tuple of Numpy arrays**: `(x_train, y_train), (x_test, y_test)`.

**`x_train`, `x_test`**: lists of sequences, which are lists of indexes
(integers). If the num_words argument was specific, the maximum
possible index value is `num_words - 1`. If the `maxlen` argument was
specified, the largest possible sequence length is `maxlen`.

**`y_train`, `y_test`**: lists of integer labels (1 or 0).

**Note**: The 'out of vocabulary' character is only used for
words that were present in the training set but are not included
because they're not making the `num_words` cut here.
Words that were not seen in the training set but are in the test set
have simply been skipped.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/datasets/imdb.py#L143" >}}

### `get_word_index` function

```python
keras.datasets.imdb.get_word_index(path="imdb_word_index.json")
```

Retrieves a dict mapping words to their index in the IMDB dataset.

**Arguments**

- **path**: where to cache the data (relative to `~/.keras/dataset`).

**Returns**

The word index dictionary. Keys are word strings, values are their
index.

**Example**

```python
start_char = 1
oov_char = 2
index_from = 3
(x_train, _), _ = keras.datasets.imdb.load_data(
    start_char=start_char, oov_char=oov_char, index_from=index_from
)
word_index = keras.datasets.imdb.get_word_index()
inverted_word_index = dict(
    (i + index_from, word) for (word, i) in word_index.items()
)
inverted_word_index[start_char] = "[START]"
inverted_word_index[oov_char] = "[OOV]"
decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])
```
