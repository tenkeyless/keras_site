---
title: Model training APIs
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/trainers/trainer.py#L38" >}}

### `compile` method

```python
Model.compile(
    optimizer="rmsprop",
    loss=None,
    loss_weights=None,
    metrics=None,
    weighted_metrics=None,
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile="auto",
    auto_scale_loss=True,
)
```

Configures the model for training.

**Example**

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.FalseNegatives(),
    ],
)
```

**Arguments**

- **optimizer**: String (name of optimizer) or optimizer instance. See `keras.optimizers`.
- **loss**: Loss function. May be a string (name of loss function), or a [`keras.losses.Loss`]({{< relref "/docs/api/losses#loss-class" >}}) instance. See `keras.losses`. A loss function is any callable with the signature `loss = fn(y_true, y_pred)`, where `y_true` are the ground truth values, and `y_pred` are the model's predictions. `y_true` should have shape `(batch_size, d0, .. dN)` (except in the case of sparse loss functions such as sparse categorical crossentropy which expects integer arrays of shape `(batch_size, d0, .. dN-1)`). `y_pred` should have shape `(batch_size, d0, .. dN)`. The loss function should return a float tensor.
- **loss_weights**: Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the _weighted sum_ of all individual losses, weighted by the `loss_weights` coefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs. If a dict, it is expected to map output names (strings) to scalar coefficients.
- **metrics**: List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a [`keras.metrics.Metric`]({{< relref "/docs/api/metrics/base_metric#metric-class" >}}) instance. See `keras.metrics`. Typically you will use `metrics=['accuracy']`. A function is any callable with the signature `result = fn(y_true, _pred)`. To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as `metrics={'a':'accuracy', 'b':['accuracy', 'mse']}`. You can also pass a list to specify a metric or a list of metrics for each output, such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or `metrics=['accuracy', ['accuracy', 'mse']]`. When you pass the strings 'accuracy' or 'acc', we convert this to one of [`keras.metrics.BinaryAccuracy`]({{< relref "/docs/api/metrics/accuracy_metrics#binaryaccuracy-class" >}}), [`keras.metrics.CategoricalAccuracy`]({{< relref "/docs/api/metrics/accuracy_metrics#categoricalaccuracy-class" >}}), [`keras.metrics.SparseCategoricalAccuracy`]({{< relref "/docs/api/metrics/accuracy_metrics#sparsecategoricalaccuracy-class" >}}) based on the shapes of the targets and of the model output. A similar conversion is done for the strings `"crossentropy"` and `"ce"` as well. The metrics passed here are evaluated without sample weighting; if you would like sample weighting to apply, you can specify your metrics via the `weighted_metrics` argument instead.
- **weighted_metrics**: List of metrics to be evaluated and weighted by `sample_weight` or `class_weight` during training and testing.
- **run_eagerly**: Bool. If `True`, this model's forward pass will never be compiled. It is recommended to leave this as `False` when training (for best performance), and to set it to `True` when debugging.
- **steps_per_execution**: Int. The number of batches to run during each a single compiled function call. Running multiple batches inside a single compiled function call can greatly improve performance on TPUs or small models with a large Python overhead. At most, one full epoch will be run each execution. If a number larger than the size of the epoch is passed, the execution will be truncated to the size of the epoch. Note that if `steps_per_execution` is set to `N`, `Callback.on_batch_begin` and `Callback.on_batch_end` methods will only be called every `N` batches (i.e. before/after each compiled function execution). Not supported with the PyTorch backend.
- **jit_compile**: Bool or `"auto"`. Whether to use XLA compilation when compiling a model. For `jax` and `tensorflow` backends, `jit_compile="auto"` enables XLA compilation if the model supports it, and disabled otherwise. For `torch` backend, `"auto"` will default to eager execution and `jit_compile=True` will run with `torch.compile` with the `"inductor"` backend.
- **auto_scale_loss**: Bool. If `True` and the model dtype policy is `"mixed_float16"`, the passed optimizer will be automatically wrapped in a `LossScaleOptimizer`, which will dynamically scale the loss to prevent underflow.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L240" >}}

### `fit` method

```python
Model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
)
```

Trains the model for a fixed number of epochs (dataset iterations).

**Arguments**

- **x**: Input data. It could be:
  - A NumPy array (or array-like), or a list of arrays (in case the model has multiple inputs).
  - A tensor, or a list of tensors (in case the model has multiple inputs).
  - A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
  - A [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). Should return a tuple of either `(inputs, targets)` or `(inputs, targets, sample_weights)`.
  - A [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) returning `(inputs, targets)` or `(inputs, targets, sample_weights)`.
- **y**: Target data. Like the input data `x`, it could be either NumPy array(s) or backend-native tensor(s). If `x` is a dataset, generator, or [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) instance, `y` should not be specified (since targets will be obtained from `x`).
- **batch_size**: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 32. Do not specify the `batch_size` if your data is in the form of datasets, generators, or [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) instances (since they generate batches).
- **epochs**: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided (unless the `steps_per_epoch` flag is set to something other than None). Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
- **verbose**: `"auto"`, 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. "auto" becomes 1 for most cases. Note that the progress bar is not particularly useful when logged to a file, so `verbose=2` is recommended when not running interactively (e.g., in a production environment). Defaults to `"auto"`.
- **callbacks**: List of [`keras.callbacks.Callback`]({{< relref "/docs/api/callbacks/base_callback#callback-class" >}}) instances. List of callbacks to apply during training. See `keras.callbacks`. Note [`keras.callbacks.ProgbarLogger`]({{< relref "/docs/api/callbacks/progbar_logger#progbarlogger-class" >}}) and `keras.callbacks.History` callbacks are created automatically and need not be passed to `model.fit()`. [`keras.callbacks.ProgbarLogger`]({{< relref "/docs/api/callbacks/progbar_logger#progbarlogger-class" >}}) is created or not based on the `verbose` argument in `model.fit()`.
- **validation_split**: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling. This argument is not supported when `x` is a dataset, generator or [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) instance. If both `validation_data` and `validation_split` are provided, `validation_data` will override `validation_split`.
- **validation_data**: Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. Thus, note the fact that the validation loss of data provided using `validation_split` or `validation_data` is not affected by regularization layers like noise and dropout. `validation_data` will override `validation_split`. It could be:
  - A tuple `(x_val, y_val)` of NumPy arrays or tensors.
  - A tuple `(x_val, y_val, val_sample_weights)` of NumPy arrays.
  - A [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
  - A Python generator or [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) returning `(inputs, targets)` or `(inputs, targets, sample_weights)`.
- **shuffle**: Boolean, whether to shuffle the training data before each epoch. This argument is ignored when `x` is a generator or a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
- **class_weight**: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class. When `class_weight` is specified and targets have a rank of 2 or greater, either `y` must be one-hot encoded, or an explicit final dimension of `1` must be included for sparse class labels.
- **sample_weight**: Optional NumPy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) NumPy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape `(samples, sequence_length)`, to apply a different weight to every timestep of every sample. This argument is not supported when `x` is a dataset, generator, or [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) instance, instead provide the sample_weights as the third element of `x`. Note that sample weighting does not apply to metrics specified via the `metrics` argument in `compile()`. To apply sample weighting to your metrics, you can specify them via the `weighted_metrics` in `compile()` instead.
- **initial_epoch**: Integer. Epoch at which to start training (useful for resuming a previous training run).
- **steps_per_epoch**: Integer or `None`. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as backend-native tensors, the default `None` is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. If `x` is a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), and `steps_per_epoch` is `None`, the epoch will run until the input dataset is exhausted. When passing an infinitely repeating dataset, you must specify the `steps_per_epoch` argument. If `steps_per_epoch=-1` the training will run indefinitely with an infinitely repeating dataset.
- **validation_steps**: Only relevant if `validation_data` is provided. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch. If `validation_steps` is `None`, validation will run until the `validation_data` dataset is exhausted. In the case of an infinitely repeated dataset, it will run into an infinite loop. If `validation_steps` is specified and only part of the dataset will be consumed, the evaluation will start from the beginning of the dataset at each epoch. This ensures that the same validation samples are used every time.
- **validation_batch_size**: Integer or `None`. Number of samples per validation batch. If unspecified, will default to `batch_size`. Do not specify the `validation_batch_size` if your data is in the form of datasets or [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) instances (since they generate batches).
- **validation_freq**: Only relevant if validation data is provided. Specifies how many training epochs to run before a new validation run is performed, e.g. `validation_freq=2` runs validation every 2 epochs.

Unpacking behavior for iterator-like inputs: A common pattern is to pass an iterator like object such as a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) or a [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) to `fit()`, which will in fact yield not only features (`x`) but optionally targets (`y`) and sample weights (`sample_weight`). Keras requires that the output of such iterator-likes be unambiguous. The iterator should return a tuple of length 1, 2, or 3, where the optional second and third elements will be used for `y` and `sample_weight` respectively. Any other type provided will be wrapped in a length-one tuple, effectively treating everything as `x`. When yielding dicts, they should still adhere to the top-level tuple structure, e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate features, targets, and weights from the keys of a single dict. A notable unsupported data type is the `namedtuple`. The reason is that it behaves like both an ordered datatype (tuple) and a mapping datatype (dict). So given a namedtuple of the form: `namedtuple("example_tuple", ["y", "x"])` it is ambiguous whether to reverse the order of the elements when interpreting the value. Even worse is a tuple of the form: `namedtuple("other_tuple", ["x", "y", "z"])` where it is unclear if the tuple was intended to be unpacked into `x`, `y`, and `sample_weight` or passed through as a single element to `x`.

**Returns**

A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L376" >}}

### `evaluate` method

```python
Model.evaluate(
    x=None,
    y=None,
    batch_size=None,
    verbose="auto",
    sample_weight=None,
    steps=None,
    callbacks=None,
    return_dict=False,
    **kwargs
)
```

Returns the loss value & metrics values for the model in test mode.

Computation is done in batches (see the `batch_size` arg.)

**Arguments**

- **x**: Input data. It could be:
  - A NumPy array (or array-like), or a list of arrays (in case the model has multiple inputs).
  - A tensor, or a list of tensors (in case the model has multiple inputs).
  - A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
  - A [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). Should return a tuple of either `(inputs, targets)` or `(inputs, targets, sample_weights)`.
  - A generator or [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) returning `(inputs, targets)` or `(inputs, targets, sample_weights)`.
- **y**: Target data. Like the input data `x`, it could be either NumPy array(s) or backend-native tensor(s). If `x` is a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) or [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) instance, `y` should not be specified (since targets will be obtained from the iterator/dataset).
- **batch_size**: Integer or `None`. Number of samples per batch of computation. If unspecified, `batch_size` will default to 32. Do not specify the `batch_size` if your data is in the form of a dataset, generators, or [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) instances (since they generate batches).
- **verbose**: `"auto"`, 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line. `"auto"` becomes 1 for most cases. Note that the progress bar is not particularly useful when logged to a file, so `verbose=2` is recommended when not running interactively (e.g. in a production environment). Defaults to `"auto"`.
- **sample_weight**: Optional NumPy array of weights for the test samples, used for weighting the loss function. You can either pass a flat (1D) NumPy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape `(samples, sequence_length)`, to apply a different weight to every timestep of every sample. This argument is not supported when `x` is a dataset, instead pass sample weights as the third element of `x`.
- **steps**: Integer or `None`. Total number of steps (batches of samples) before declaring the evaluation round finished. Ignored with the default value of `None`. If `x` is a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) and `steps` is `None`, evaluation will run until the dataset is exhausted.
- **callbacks**: List of [`keras.callbacks.Callback`]({{< relref "/docs/api/callbacks/base_callback#callback-class" >}}) instances. List of callbacks to apply during evaluation.
- **return_dict**: If `True`, loss and metric results are returned as a dict, with each key being the name of the metric. If `False`, they are returned as a list.

**Returns**

Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute `model.metrics_names` will give you the display labels for the scalar outputs.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L443" >}}

### `predict` method

```python
Model.predict(x, batch_size=None, verbose="auto", steps=None, callbacks=None)
```

Generates output predictions for the input samples.

Computation is done in batches. This method is designed for batch processing of large numbers of inputs. It is not intended for use inside of loops that iterate over your data and process small numbers of inputs at a time.

For small numbers of inputs that fit in one batch, directly use `__call__()` for faster execution, e.g., `model(x)`, or `model(x, training=False)` if you have layers such as `BatchNormalization` that behave differently during inference.

Note: See [this FAQ entry]({{< relref "/docs/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-__call__" >}}) for more details about the difference between `Model` methods `predict()` and `__call__()`.

**Arguments**

- **x**: Input samples. It could be:
  - A NumPy array (or array-like), or a list of arrays (in case the model has multiple inputs).
  - A tensor, or a list of tensors (in case the model has multiple inputs).
  - A [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
  - A [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) instance.
- **batch_size**: Integer or `None`. Number of samples per batch. If unspecified, `batch_size` will default to 32. Do not specify the `batch_size` if your data is in the form of dataset, generators, or [`keras.utils.PyDataset`]({{< relref "/docs/api/utils/python_utils#pydataset-class" >}}) instances (since they generate batches).
- **verbose**: `"auto"`, 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line. `"auto"` becomes 1 for most cases. Note that the progress bar is not particularly useful when logged to a file, so `verbose=2` is recommended when not running interactively (e.g. in a production environment). Defaults to `"auto"`.
- **steps**: Total number of steps (batches of samples) before declaring the prediction round finished. Ignored with the default value of `None`. If `x` is a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) and `steps` is `None`, `predict()` will run until the input dataset is exhausted.
- **callbacks**: List of [`keras.callbacks.Callback`]({{< relref "/docs/api/callbacks/base_callback#callback-class" >}}) instances. List of callbacks to apply during prediction.

**Returns**

NumPy array(s) of predictions.

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L521" >}}

### `train_on_batch` method

```python
Model.train_on_batch(
    x, y=None, sample_weight=None, class_weight=None, return_dict=False
)
```

Runs a single gradient update on a single batch of data.

**Arguments**

- **x**: Input data. Must be array-like.
- **y**: Target data. Must be array-like.
- **sample_weight**: Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape `(samples, sequence_length)`, to apply a different weight to every timestep of every sample.
- **class_weight**: Optional dictionary mapping class indices (integers) to a weight (float) to apply to the model's loss for the samples from this class during training. This can be useful to tell the model to "pay more attention" to samples from an under-represented class. When `class_weight` is specified and targets have a rank of 2 or greater, either `y` must be one-hot encoded, or an explicit final dimension of 1 must be included for sparse class labels.
- **return_dict**: If `True`, loss and metric results are returned as a dict, with each key being the name of the metric. If `False`, they are returned as a list.

**Returns**

A scalar loss value (when no metrics and `return_dict=False`), a list of loss and metric values (if there are metrics and `return_dict=False`), or a dict of metric and loss values (if `return_dict=True`).

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L555" >}}

### `test_on_batch` method

```python
Model.test_on_batch(x, y=None, sample_weight=None, return_dict=False)
```

Test the model on a single batch of samples.

**Arguments**

- **x**: Input data. Must be array-like.
- **y**: Target data. Must be array-like.
- **sample_weight**: Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape `(samples, sequence_length)`, to apply a different weight to every timestep of every sample.
- **return_dict**: If `True`, loss and metric results are returned as a dict, with each key being the name of the metric. If `False`, they are returned as a list.

**Returns**

A scalar loss value (when no metrics and `return_dict=False`), a list of loss and metric values (if there are metrics and `return_dict=False`), or a dict of metric and loss values (if `return_dict=True`).

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L577" >}}

### `predict_on_batch` method

```python
Model.predict_on_batch(x)
```

Returns predictions for a single batch of samples.

**Arguments**

- **x**: Input data. It must be array-like.

**Returns**

NumPy array(s) of predictions.
