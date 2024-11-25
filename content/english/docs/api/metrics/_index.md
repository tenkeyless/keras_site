---
title: Metrics
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

A metric is a function that is used to judge the performance of your model.

Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model.
Note that you may use any loss function as a metric.

## Available metrics

### [Base Metric class]({{< relref "/docs/api/metrics/base_metric/" >}})

- [Metric class]({{< relref "/docs/api/metrics/base_metric/#metric-class" >}})

### [Accuracy metrics]({{< relref "/docs/api/metrics/accuracy_metrics/" >}})

- [Accuracy class]({{< relref "/docs/api/metrics/accuracy_metrics/#accuracy-class" >}})
- [BinaryAccuracy class]({{< relref "/docs/api/metrics/accuracy_metrics/#binaryaccuracy-class" >}})
- [CategoricalAccuracy class]({{< relref "/docs/api/metrics/accuracy_metrics/#categoricalaccuracy-class" >}})
- [SparseCategoricalAccuracy class]({{< relref "/docs/api/metrics/accuracy_metrics/#sparsecategoricalaccuracy-class" >}})
- [TopKCategoricalAccuracy class]({{< relref "/docs/api/metrics/accuracy_metrics/#topkcategoricalaccuracy-class" >}})
- [SparseTopKCategoricalAccuracy class]({{< relref "/docs/api/metrics/accuracy_metrics/#sparsetopkcategoricalaccuracy-class" >}})

### [Probabilistic metrics]({{< relref "/docs/api/metrics/probabilistic_metrics/" >}})

- [BinaryCrossentropy class]({{< relref "/docs/api/metrics/probabilistic_metrics/#binarycrossentropy-class" >}})
- [CategoricalCrossentropy class]({{< relref "/docs/api/metrics/probabilistic_metrics/#categoricalcrossentropy-class" >}})
- [SparseCategoricalCrossentropy class]({{< relref "/docs/api/metrics/probabilistic_metrics/#sparsecategoricalcrossentropy-class" >}})
- [KLDivergence class]({{< relref "/docs/api/metrics/probabilistic_metrics/#kldivergence-class" >}})
- [Poisson class]({{< relref "/docs/api/metrics/probabilistic_metrics/#poisson-class" >}})

### [Regression metrics]({{< relref "/docs/api/metrics/regression_metrics/" >}})

- [MeanSquaredError class]({{< relref "/docs/api/metrics/regression_metrics/#meansquarederror-class" >}})
- [RootMeanSquaredError class]({{< relref "/docs/api/metrics/regression_metrics/#rootmeansquarederror-class" >}})
- [MeanAbsoluteError class]({{< relref "/docs/api/metrics/regression_metrics/#meanabsoluteerror-class" >}})
- [MeanAbsolutePercentageError class]({{< relref "/docs/api/metrics/regression_metrics/#meanabsolutepercentageerror-class" >}})
- [MeanSquaredLogarithmicError class]({{< relref "/docs/api/metrics/regression_metrics/#meansquaredlogarithmicerror-class" >}})
- [CosineSimilarity class]({{< relref "/docs/api/metrics/regression_metrics/#cosinesimilarity-class" >}})
- [LogCoshError class]({{< relref "/docs/api/metrics/regression_metrics/#logcosherror-class" >}})
- [R2Score class]({{< relref "/docs/api/metrics/regression_metrics/#r2score-class" >}})

### [Classification metrics based on True/False positives & negatives]({{< relref "/docs/api/metrics/classification_metrics/" >}})

- [AUC class]({{< relref "/docs/api/metrics/classification_metrics/#auc-class" >}})
- [Precision class]({{< relref "/docs/api/metrics/classification_metrics/#precision-class" >}})
- [Recall class]({{< relref "/docs/api/metrics/classification_metrics/#recall-class" >}})
- [TruePositives class]({{< relref "/docs/api/metrics/classification_metrics/#truepositives-class" >}})
- [TrueNegatives class]({{< relref "/docs/api/metrics/classification_metrics/#truenegatives-class" >}})
- [FalsePositives class]({{< relref "/docs/api/metrics/classification_metrics/#falsepositives-class" >}})
- [FalseNegatives class]({{< relref "/docs/api/metrics/classification_metrics/#falsenegatives-class" >}})
- [PrecisionAtRecall class]({{< relref "/docs/api/metrics/classification_metrics/#precisionatrecall-class" >}})
- [RecallAtPrecision class]({{< relref "/docs/api/metrics/classification_metrics/#recallatprecision-class" >}})
- [SensitivityAtSpecificity class]({{< relref "/docs/api/metrics/classification_metrics/#sensitivityatspecificity-class" >}})
- [SpecificityAtSensitivity class]({{< relref "/docs/api/metrics/classification_metrics/#specificityatsensitivity-class" >}})
- [F1Score class]({{< relref "/docs/api/metrics/classification_metrics/#f1score-class" >}})
- [FBetaScore class]({{< relref "/docs/api/metrics/classification_metrics/#fbetascore-class" >}})

### [Image segmentation metrics]({{< relref "/docs/api/metrics/segmentation_metrics/" >}})

- [IoU class]({{< relref "/docs/api/metrics/segmentation_metrics/#iou-class" >}})
- [BinaryIoU class]({{< relref "/docs/api/metrics/segmentation_metrics/#binaryiou-class" >}})
- [OneHotIoU class]({{< relref "/docs/api/metrics/segmentation_metrics/#onehotiou-class" >}})
- [OneHotMeanIoU class]({{< relref "/docs/api/metrics/segmentation_metrics/#onehotmeaniou-class" >}})
- [MeanIoU class]({{< relref "/docs/api/metrics/segmentation_metrics/#meaniou-class" >}})

### [Hinge metrics for "maximum-margin" classification]({{< relref "/docs/api/metrics/hinge_metrics/" >}})

- [Hinge class]({{< relref "/docs/api/metrics/hinge_metrics/#hinge-class" >}})
- [SquaredHinge class]({{< relref "/docs/api/metrics/hinge_metrics/#squaredhinge-class" >}})
- [CategoricalHinge class]({{< relref "/docs/api/metrics/hinge_metrics/#categoricalhinge-class" >}})

### [Metric wrappers and reduction metrics]({{< relref "/docs/api/metrics/metrics_wrappers/" >}})

- [MeanMetricWrapper class]({{< relref "/docs/api/metrics/metrics_wrappers/#meanmetricwrapper-class" >}})
- [Mean class]({{< relref "/docs/api/metrics/metrics_wrappers/#mean-class" >}})
- [Sum class]({{< relref "/docs/api/metrics/metrics_wrappers/#sum-class" >}})

## Usage with `compile()` & `fit()`

The `compile()` method takes a `metrics` argument, which is a list of metrics:

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[
        metrics.MeanSquaredError(),
        metrics.AUC(),
    ]
)
```

Metric values are displayed during `fit()` and logged to the `History` object returned
by `fit()`. They are also returned by `model.evaluate()`.

Note that the best way to monitor your metrics during training is via [TensorBoard]({{< relref "/docs/api/callbacks/tensorboard" >}}).

To track metrics under a specific name, you can pass the `name` argument
to the metric constructor:

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[
        metrics.MeanSquaredError(name='my_mse'),
        metrics.AUC(name='my_auc'),
    ]
)
```

All built-in metrics may also be passed via their string identifier (in this case,
default constructor argument values are used, including a default metric name):

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[
        'MeanSquaredError',
        'AUC',
    ]
)
```

## Standalone usage

Unlike losses, metrics are stateful. You update their state using the `update_state()` method,
and you query the scalar metric result using the `result()` method:

```python
m = keras.metrics.AUC()
m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
print('Intermediate result:', float(m.result()))
m.update_state([1, 1, 1, 1], [0, 1, 1, 0])
print('Final result:', float(m.result()))
```

The internal state can be cleared via `metric.reset_states()`.

Here's how you would use a metric as part of a simple custom training loop:

```python
accuracy = keras.metrics.CategoricalAccuracy()
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()
for step, (x, y) in enumerate(dataset):
    with tf.GradientTape() as tape:
        logits = model(x)
        # Compute the loss value for this batch.
        loss_value = loss_fn(y, logits)
    # Update the state of the `accuracy` metric.
    accuracy.update_state(y, logits)
    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    # Logging the current accuracy value so far.
    if step % 100 == 0:
        print('Step:', step)
        print('Total running accuracy so far: %.3f' % accuracy.result())
```

## Creating custom metrics

### As simple callables (stateless)

Much like loss functions, any callable with signature `metric_fn(y_true, y_pred)`
that returns an array of losses (one of sample in the input batch) can be passed to `compile()` as a metric.
Note that sample weighting is automatically supported for any such metric.

Here's a simple example:

```python
from keras import ops
def my_metric_fn(y_true, y_pred):
    squared_difference = ops.square(y_true - y_pred)
    return ops.mean(squared_difference, axis=-1)  # Note the `axis=-1`
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[my_metric_fn])
```

In this case, the scalar metric value you are tracking during training and evaluation
is the average of the per-batch metric values for all batches see during a given epoch
(or during a given call to `model.evaluate()`).

### As subclasses of `Metric` (stateful)

Not all metrics can be expressed via stateless callables, because
metrics are evaluated for each batch during training and evaluation, but in some cases
the average of the per-batch values is not what you are interested in.

Let's say that you want to compute AUC over a
given evaluation dataset: the average of the per-batch AUC values
isn't the same as the AUC over the entire dataset.

For such metrics, you're going to want to subclass the `Metric` class,
which can maintain a state across batches. It's easy:

- Create the state variables in `__init__`
- Update the variables given `y_true` and `y_pred` in `update_state()`
- Return the scalar metric result in `result()`
- Clear the state in `reset_states()`

Here's a simple example computing binary true positives:

```python
class BinaryTruePositives(keras.metrics.Metric):
  def __init__(self, name='binary_true_positives', **kwargs):
    super().__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = ops.cast(y_true, "bool")
    y_pred = ops.cast(y_pred, "bool")
    values = ops.logical_and(ops.equal(y_true, True), ops.equal(y_pred, True))
    values = ops.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = ops.cast(sample_weight, self.dtype)
      values = values * sample_weight
    self.true_positives.assign_add(ops.sum(values))
  def result(self):
    return self.true_positives
  def reset_state(self):
    self.true_positives.assign(0)
m = BinaryTruePositives()
m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
print(f'Intermediate result: {m.result().numpy()}')
m.update_state([1, 1, 1, 1], [0, 1, 1, 0])
print(f'Intermediate result: {m.result().numpy()}')
```
