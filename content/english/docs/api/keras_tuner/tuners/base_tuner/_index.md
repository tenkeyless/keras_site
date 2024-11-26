---
title: The base Tuner class
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/tuner.py#L35" >}}

### `Tuner` class

```python
keras_tuner.Tuner(
    oracle,
    hypermodel=None,
    max_model_size=None,
    optimizer=None,
    loss=None,
    metrics=None,
    distribution_strategy=None,
    directory=None,
    project_name=None,
    logger=None,
    tuner_id=None,
    overwrite=False,
    executions_per_trial=1,
    **kwargs
)
```

Tuner class for Keras models.

This is the base `Tuner` class for all tuners for Keras models. It manages
the building, training, evaluation and saving of the Keras models. New
tuners can be created by subclassing the class.

All Keras related logics are in `Tuner.run_trial()` and its subroutines.
When subclassing `Tuner`, if not calling `super().run_trial()`, it can tune
anything.

**Arguments**

- **oracle**: Instance of `Oracle` class.
- **hypermodel**: Instance of `HyperModel` class (or callable that takes
  hyperparameters and returns a `Model` instance). It is optional
  when `Tuner.run_trial()` is overriden and does not use
  `self.hypermodel`.
- **max_model_size**: Integer, maximum number of scalars in the parameters of
  a model. Models larger than this are rejected.
- **optimizer**: Optional optimizer. It is used to override the `optimizer`
  argument in the `compile` step for the models. If the hypermodel
  does not compile the models it generates, then this argument must be
  specified.
- **loss**: Optional loss. May be used to override the `loss` argument in the
  `compile` step for the models. If the hypermodel does not compile
  the models it generates, then this argument must be specified.
- **metrics**: Optional metrics. May be used to override the `metrics`
  argument in the `compile` step for the models. If the hypermodel
  does not compile the models it generates, then this argument must
  be specified.
- **distribution_strategy**: Optional instance of [`tf.distribute.Strategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy).
  If specified, each trial will run under this scope. For example,
  `tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])` will run
  each trial on two GPUs. Currently only single-worker strategies are
  supported.
- **directory**: A string, the relative path to the working directory.
- **project_name**: A string, the name to use as prefix for files saved by
  this `Tuner`.
- **tuner_id**: Optional string, used as the ID of this `Tuner`.
- **overwrite**: Boolean, defaults to `False`. If `False`, reloads an
  existing project of the same name if one is found. Otherwise,
  overwrites the project.
- **executions_per_trial**: Integer, the number of executions (training a
  model from scratch, starting from a new initialization) to run per
  trial (model configuration). Model metrics may vary greatly
  depending on random initialization, hence it is often a good idea
  to run several executions per trial in order to evaluate the
  performance of a given set of hyperparameter values.
- **\*\*kwargs**: Arguments for `BaseTuner`.

**Attributes**

- **remaining_trials**: Number of trials remaining, `None` if `max_trials` is
  not set. This is useful when resuming a previously stopped search.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/base_tuner.py#L369" >}}

### `get_best_hyperparameters` method

```python
Tuner.get_best_hyperparameters(num_trials=1)
```

Returns the best hyperparameters, as determined by the objective.

This method can be used to reinstantiate the (untrained) best model
found during the search process.

**Example**

```python
best_hp = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hp)
```

**Arguments**

- **num_trials**: Optional number of `HyperParameters` objects to return.

**Returns**

List of `HyperParameter` objects sorted from the best to the worst.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/tuner.py#L381" >}}

### `get_best_models` method

```python
Tuner.get_best_models(num_models=1)
```

Returns the best model(s), as determined by the tuner's objective.

The models are loaded with the weights corresponding to
their best checkpoint (at the end of the best epoch of best trial).

This method is for querying the models trained during the search.
For best performance, it is recommended to retrain your Model on the
full dataset using the best hyperparameters found during `search`,
which can be obtained using `tuner.get_best_hyperparameters()`.

**Arguments**

- **num_models**: Optional number of best models to return.
  Defaults to 1.

**Returns**

List of trained model instances sorted from the best to the worst.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/base_tuner.py#L439" >}}

### `get_state` method

```python
Tuner.get_state()
```

Returns the current state of this object.

This method is called during `save`.

**Returns**

A dictionary of serializable objects as the state.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/tuner.py#L319" >}}

### `load_model` method

```python
Tuner.load_model(trial)
```

Loads a Model from a given trial.

For models that report intermediate results to the `Oracle`, generally
`load_model` should load the best reported `step` by relying of
`trial.best_step`.

**Arguments**

- **trial**: A `Trial` instance, the `Trial` corresponding to the model
  to load.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/tuner.py#L356" >}}

### `on_epoch_begin` method

```python
Tuner.on_epoch_begin(trial, model, epoch, logs=None)
```

Called at the beginning of an epoch.

**Arguments**

- **trial**: A `Trial` instance.
- **model**: A Keras `Model`.
- **epoch**: The current epoch number.
- **logs**: Additional metrics.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/tuner.py#L334" >}}

### `on_batch_begin` method

```python
Tuner.on_batch_begin(trial, model, batch, logs)
```

Called at the beginning of a batch.

**Arguments**

- **trial**: A `Trial` instance.
- **model**: A Keras `Model`.
- **batch**: The current batch number within the current epoch.
- **logs**: Additional metrics.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/tuner.py#L345" >}}

### `on_batch_end` method

```python
Tuner.on_batch_end(trial, model, batch, logs=None)
```

Called at the end of a batch.

**Arguments**

- **trial**: A `Trial` instance.
- **model**: A Keras `Model`.
- **batch**: The current batch number within the current epoch.
- **logs**: Additional metrics.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/tuner.py#L367" >}}

### `on_epoch_end` method

```python
Tuner.on_epoch_end(trial, model, epoch, logs=None)
```

Called at the end of an epoch.

**Arguments**

- **trial**: A `Trial` instance.
- **model**: A Keras `Model`.
- **epoch**: The current epoch number.
- **logs**: Dict. Metrics for this epoch. This should include
  the value of the objective for this epoch.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/tuner.py#L247" >}}

### `run_trial` method

```python
Tuner.run_trial(trial, )
```

Evaluates a set of hyperparameter values.

This method is called multiple times during `search` to build and
evaluate the models with different hyperparameters and return the
objective value.

**Example**

You can use it with `self.hypermodel` to build and fit the model.

```python
def run_trial(self, trial, *args, **kwargs):
    hp = trial.hyperparameters
    model = self.hypermodel.build(hp)
    return self.hypermodel.fit(hp, model, *args, **kwargs)
```

You can also use it as a black-box optimizer for anything.

```python
def run_trial(self, trial, *args, **kwargs):
    hp = trial.hyperparameters
    x = hp.Float("x", -2.0, 2.0)
    y = x * x + 2 * x + 1
    return y
```

**Arguments**

- **trial**: A `Trial` instance that contains the information needed to
  run this trial. Hyperparameters can be accessed via
  `trial.hyperparameters`.
- **\*args**: Positional arguments passed by `search`.
- **\*\*kwargs**: Keyword arguments passed by `search`.

**Returns**

A `History` object, which is the return value of `model.fit()`, a
dictionary, a float, or a list of one of these types.

If return a dictionary, it should be a dictionary of the metrics to
track. The keys are the metric names, which contains the
`objective` name. The values should be the metric values.

If return a float, it should be the `objective` value.

If evaluating the model for multiple times, you may return a list
of results of any of the types above. The final objective value is
the average of the results in the list.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/base_tuner.py#L411" >}}

### `results_summary` method

```python
Tuner.results_summary(num_trials=10)
```

Display tuning results summary.

The method prints a summary of the search results including the
hyperparameter values and evaluation results for each trial.

**Arguments**

- **num_trials**: Optional number of trials to display. Defaults to 10.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/base_tuner.py#L297" >}}

### `save_model` method

```python
Tuner.save_model(trial_id, model, step=0)
```

Saves a Model for a given trial.

**Arguments**

- **trial_id**: The ID of the `Trial` corresponding to this Model.
- **model**: The trained model.
- **step**: Integer, for models that report intermediate results to the
  `Oracle`, the step the saved file correspond to. For example,
  for Keras models this is the number of epochs trained.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/base_tuner.py#L194" >}}

### `search` method

```python
Tuner.search(*fit_args, **fit_kwargs)
```

Performs a search for best hyperparameter configuations.

**Arguments**

- **\*fit_args**: Positional arguments that should be passed to
  `run_trial`, for example the training and validation data.
- **\*\*fit_kwargs**: Keyword arguments that should be passed to
  `run_trial`, for example the training and validation data.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/base_tuner.py#L392" >}}

### `search_space_summary` method

```python
Tuner.search_space_summary(extended=False)
```

Print search space summary.

The methods prints a summary of the hyperparameters in the search
space, which can be called before calling the `search` method.

**Arguments**

- **extended**: Optional boolean, whether to display an extended summary.
  Defaults to False.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/base_tuner.py#L442" >}}

### `set_state` method

```python
Tuner.set_state(state)
```

Sets the current state of this object.

This method is called during `reload`.

**Arguments**

- **state**: A dictionary of serialized objects as the state to restore.
