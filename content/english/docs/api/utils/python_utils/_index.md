---
title: Python & NumPy utilities
toc: true
weight: 5
type: docs
---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/utils/rng_utils.py#L10)

### `set_random_seed` function

`keras.utils.set_random_seed(seed)`

Sets all random seeds (Python, NumPy, and backend framework, e.g. TF).

You can use this utility to make almost any Keras program fully deterministic. Some limitations apply in cases where network communications are involved (e.g. parameter server distribution), which creates additional sources of randomness, or when certain non-deterministic cuDNN ops are involved.

Calling this utility is equivalent to the following:

`import random random.seed(seed)  import numpy as np np.random.seed(seed)  import tensorflow as tf  # Only if TF is installed tf.random.set_seed(seed)  import torch  # Only if the backend is 'torch' torch.manual_seed(seed)`

Note that the TensorFlow seed is set even if you're not using TensorFlow as your backend framework, since many workflows leverage [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) pipelines (which feature random shuffling). Likewise many workflows might leverage NumPy APIs.

**Arguments**

- **seed**: Integer, the random seed to use.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/utils/dataset_utils.py#L15)

### `split_dataset` function

`keras.utils.split_dataset(     dataset, left_size=None, right_size=None, shuffle=False, seed=None )`

Splits a dataset into a left half and a right half (e.g. train / test).

**Arguments**

- **dataset**: A [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), a `torch.utils.data.Dataset` object, or a list/tuple of arrays with the same length.
- **left_size**: If float (in the range `[0, 1]`), it signifies the fraction of the data to pack in the left dataset. If integer, it signifies the number of samples to pack in the left dataset. If `None`, defaults to the complement to `right_size`. Defaults to `None`.
- **right_size**: If float (in the range `[0, 1]`), it signifies the fraction of the data to pack in the right dataset. If integer, it signifies the number of samples to pack in the right dataset. If `None`, defaults to the complement to `left_size`. Defaults to `None`.
- **shuffle**: Boolean, whether to shuffle the data before splitting it.
- **seed**: A random seed for shuffling.

**Returns**

- **A tuple of two [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) objects**: the left and right splits.

**Example**

`>>> data = np.random.random(size=(1000, 4)) >>> left_ds, right_ds = keras.utils.split_dataset(data, left_size=0.8) >>> int(left_ds.cardinality()) 800 >>> int(right_ds.cardinality()) 200`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/trainers/data_adapters/data_adapter_utils.py#L53)

### `pack_x_y_sample_weight` function

`keras.utils.pack_x_y_sample_weight(x, y=None, sample_weight=None)`

Packs user-provided data into a tuple.

This is a convenience utility for packing data into the tuple formats that `Model.fit()` uses.

**Example**

`>>> x = ops.ones((10, 1)) >>> data = pack_x_y_sample_weight(x) >>> isinstance(data, ops.Tensor) True >>> y = ops.ones((10, 1)) >>> data = pack_x_y_sample_weight(x, y) >>> isinstance(data, tuple) True >>> x, y = data`

**Arguments**

- **x**: Features to pass to `Model`.
- **y**: Ground-truth targets to pass to `Model`.
- **sample_weight**: Sample weight for each element.

**Returns**

Tuple in the format used in `Model.fit()`.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/utils/file_utils.py#L129)

### `get_file` function

`keras.utils.get_file(     fname=None,     origin=None,     untar=False,     md5_hash=None,     file_hash=None,     cache_subdir="datasets",     hash_algorithm="auto",     extract=False,     archive_format="auto",     cache_dir=None,     force_download=False, )`

Downloads a file from a URL if it not already in the cache.

By default the file at the url `origin` is downloaded to the cache_dir `~/.keras`, placed in the cache_subdir `datasets`, and given the filename `fname`. The final location of a file `example.txt` would therefore be `~/.keras/datasets/example.txt`. Files in `.tar`, `.tar.gz`, `.tar.bz`, and `.zip` formats can also be extracted.

Passing a hash will verify the file after download. The command line programs `shasum` and `sha256sum` can compute the hash.

**Example**

`path_to_downloaded_file = get_file(     origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",     extract=True, )`

**Arguments**

- **fname**: If the target is a single file, this is your desired local name for the file. If `None`, the name of the file at `origin` will be used. If downloading and extracting a directory archive, the provided `fname` will be used as extraction directory name (only if it doesn't have an extension).
- **origin**: Original URL of the file.
- **untar**: Deprecated in favor of `extract` argument. Boolean, whether the file is a tar archive that should be extracted.
- **md5_hash**: Deprecated in favor of `file_hash` argument. md5 hash of the file for file integrity verification.
- **file_hash**: The expected hash string of the file after download. The sha256 and md5 hash algorithms are both supported.
- **cache_subdir**: Subdirectory under the Keras cache dir where the file is saved. If an absolute path, e.g. `"/path/to/folder"` is specified, the file will be saved at that location.
- **hash_algorithm**: Select the hash algorithm to verify the file. options are `"md5'`, `"sha256'`, and `"auto'`. The default 'auto' detects the hash algorithm in use.
- **extract**: If `True`, extracts the archive. Only applicable to compressed archive files like tar or zip.
- **archive_format**: Archive format to try for extracting the file. Options are `"auto'`, `"tar'`, `"zip'`, and `None`. `"tar"` includes tar, tar.gz, and tar.bz files. The default `"auto"` corresponds to `["tar", "zip"]`. None or an empty list will return no matches found.
- **cache_dir**: Location to store cached files, when None it defaults ether `$KERAS_HOME` if the `KERAS_HOME` environment variable is set or `~/.keras/`.
- **force_download**: If `True`, the file will always be re-downloaded regardless of the cache state.

**Returns**

Path to the downloaded file.

**⚠️ Warning on malicious downloads ⚠️**

Downloading something from the Internet carries a risk. NEVER download a file/archive if you do not trust the source. We recommend that you specify the `file_hash` argument (if the hash of the source file is known) to make sure that the file you are getting is the one you expect.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/utils/progbar.py#L11)

### `Progbar` class

`keras.utils.Progbar(     target, width=20, verbose=1, interval=0.05, stateful_metrics=None, unit_name="step" )`

Displays a progress bar.

**Arguments**

- **target**: Total number of steps expected, None if unknown.
- **width**: Progress bar width on screen.
- **verbose**: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
- **stateful_metrics**: Iterable of string names of metrics that should _not_ be averaged over time. Metrics in this list will be displayed as-is. All others will be averaged by the progbar before display.
- **interval**: Minimum visual progress update interval (in seconds).
- **unit_name**: Display name for step counts (usually "step" or "sample").

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/trainers/data_adapters/py_dataset_adapter.py#L17)

### `PyDataset` class

`keras.utils.PyDataset(workers=1, use_multiprocessing=False, max_queue_size=10)`

Base class for defining a parallel dataset using Python code.

Every `PyDataset` must implement the `__getitem__()` and the `__len__()` methods. If you want to modify your dataset between epochs, you may additionally implement `on_epoch_end()`, or `on_epoch_begin` to be called at the start of each epoch. The `__getitem__()` method should return a complete batch (not a single sample), and the `__len__` method should return the number of batches in the dataset (rather than the number of samples).

**Arguments**

- **workers**: Number of workers to use in multithreading or multiprocessing.
- **use_multiprocessing**: Whether to use Python multiprocessing for parallelism. Setting this to `True` means that your dataset will be replicated in multiple forked processes. This is necessary to gain compute-level (rather than I/O level) benefits from parallelism. However it can only be set to `True` if your dataset can be safely pickled.
- **max_queue_size**: Maximum number of batches to keep in the queue when iterating over the dataset in a multithreaded or multiprocessed setting. Reduce this value to reduce the CPU memory consumption of your dataset. Defaults to 10.

Notes:

- `PyDataset` is a safer way to do multiprocessing. This structure guarantees that the model will only train once on each sample per epoch, which is not the case with Python generators.
- The arguments `workers`, `use_multiprocessing`, and `max_queue_size` exist to configure how `fit()` uses parallelism to iterate over the dataset. They are not being used by the `PyDataset` class directly. When you are manually iterating over a `PyDataset`, no parallelism is applied.

**Example**

`` from skimage.io import imread from skimage.transform import resize import numpy as np import math  # Here, `x_set` is list of path to the images # and `y_set` are the associated classes.  class CIFAR10PyDataset(keras.utils.PyDataset):      def __init__(self, x_set, y_set, batch_size, **kwargs):         super().__init__(**kwargs)         self.x, self.y = x_set, y_set         self.batch_size = batch_size      def __len__(self):         # Return number of batches.         return math.ceil(len(self.x) / self.batch_size)      def __getitem__(self, idx):         # Return x, y for batch idx.         low = idx * self.batch_size         # Cap upper bound at array length; the last batch may be smaller         # if the total number of items is not a multiple of batch size.         high = min(low + self.batch_size, len(self.x))         batch_x = self.x[low:high]         batch_y = self.y[low:high]          return np.array([             resize(imread(file_name), (200, 200))                for file_name in batch_x]), np.array(batch_y) ``

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/utils/numerical_utils.py#L38)

### `to_categorical` function

`keras.utils.to_categorical(x, num_classes=None)`

Converts a class vector (integers) to binary class matrix.

E.g. for use with `categorical_crossentropy`.

**Arguments**

- **x**: Array-like with class values to be converted into a matrix (integers from 0 to `num_classes - 1`).
- **num_classes**: Total number of classes. If `None`, this would be inferred as `max(x) + 1`. Defaults to `None`.

**Returns**

A binary matrix representation of the input as a NumPy array. The class axis is placed last.

**Example**

`>>> a = keras.utils.to_categorical([0, 1, 2, 3], num_classes=4) >>> print(a) [[1. 0. 0. 0.]  [0. 1. 0. 0.]  [0. 0. 1. 0.]  [0. 0. 0. 1.]]`

`>>> b = np.array([.9, .04, .03, .03, ...               .3, .45, .15, .13, ...               .04, .01, .94, .05, ...               .12, .21, .5, .17], ...               shape=[4, 4]) >>> loss = keras.ops.categorical_crossentropy(a, b) >>> print(np.around(loss, 5)) [0.10536 0.82807 0.1011  1.77196]`

`>>> loss = keras.ops.categorical_crossentropy(a, a) >>> print(np.around(loss, 5)) [0. 0. 0. 0.]`

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/utils/numerical_utils.py#L8)

### `normalize` function

`keras.utils.normalize(x, axis=-1, order=2)`

Normalizes an array.

If the input is a NumPy array, a NumPy array will be returned. If it's a backend tensor, a backend tensor will be returned.

**Arguments**

- **x**: Array to normalize.
- **axis**: axis along which to normalize.
- **order**: Normalization order (e.g. `order=2` for L2 norm).

**Returns**

A normalized copy of the array.

---
