---
title: Classification metrics based on True/False positives & negatives
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L1067" >}}

### `AUC` class

```python
keras.metrics.AUC(
    num_thresholds=200,
    curve="ROC",
    summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    num_labels=None,
    label_weights=None,
    from_logits=False,
)
```

Approximates the AUC (Area under the curve) of the ROC or PR curves.

The AUC (Area under the curve) of the ROC (Receiver operating
characteristic; default) or PR (Precision Recall) curves are quality
measures of binary classifiers. Unlike the accuracy, and like cross-entropy
losses, ROC-AUC and PR-AUC evaluate all the operational points of a model.

This class approximates AUCs using a Riemann sum. During the metric
accumulation phrase, predictions are accumulated within predefined buckets
by value. The AUC is then computed by interpolating per-bucket averages.
These buckets define the evaluated operational points.

This metric creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the AUC. To discretize the AUC curve, a linearly spaced set of
thresholds is used to compute pairs of recall and precision values. The area
under the ROC-curve is therefore computed using the height of the recall
values by the false positive rate, while the area under the PR-curve is the
computed using the height of the precision values by the recall.

This value is ultimately returned as `auc`, an idempotent operation that
computes the area under a discretized curve of precision versus recall
values (computed using the aforementioned variables). The `num_thresholds`
variable controls the degree of discretization with larger numbers of
thresholds more closely approximating the true AUC. The quality of the
approximation may vary dramatically depending on `num_thresholds`. The
`thresholds` parameter can be used to manually specify thresholds which
split the predictions more evenly.

For a best approximation of the real AUC, `predictions` should be
distributed approximately uniformly in the range `[0, 1]` (if
`from_logits=False`). The quality of the AUC approximation may be poor if
this is not the case. Setting `summation_method` to 'minoring' or 'majoring'
can help quantify the error in the approximation by providing lower or upper
bound estimate of the AUC.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

**Arguments**

- **num_thresholds**: (Optional) The number of thresholds to
  use when discretizing the roc curve. Values must be > 1.
  Defaults to `200`.
- **curve**: (Optional) Specifies the name of the curve to be computed,
  `'ROC'` (default) or `'PR'` for the Precision-Recall-curve.
- **summation_method**: (Optional) Specifies the [Riemann summation method](https://en.wikipedia.org/wiki/Riemann_sum) used.
  'interpolation' (default) applies mid-point summation scheme for
  `ROC`. For PR-AUC, interpolates (true/false) positives but not
  the ratio that is precision (see Davis & Goadrich 2006 for
  details); 'minoring' applies left summation for increasing
  intervals and right summation for decreasing intervals; 'majoring'
  does the opposite.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.
- **thresholds**: (Optional) A list of floating point values to use as the
  thresholds for discretizing the curve. If set, the `num_thresholds`
  parameter is ignored. Values should be in `[0, 1]`. Endpoint
  thresholds equal to {`-epsilon`, `1+epsilon`} for a small positive
  epsilon value will be automatically included with these to correctly
  handle predictions equal to exactly 0 or 1.
- **multi_label**: boolean indicating whether multilabel data should be
  treated as such, wherein AUC is computed separately for each label
  and then averaged across labels, or (when `False`) if the data
  should be flattened into a single label before AUC computation. In
  the latter case, when multilabel data is passed to AUC, each
  label-prediction pair is treated as an individual data point. Should
  be set to `False` for multi-class data.
- **num_labels**: (Optional) The number of labels, used when `multi_label` is
  True. If `num_labels` is not specified, then state variables get
  created on the first call to `update_state`.
- **label_weights**: (Optional) list, array, or tensor of non-negative weights
  used to compute AUCs for multilabel data. When `multi_label` is
  True, the weights are applied to the individual label AUCs when they
  are averaged to produce the multi-label AUC. When it's False, they
  are used to weight the individual label predictions in computing the
  confusion matrix on the flattened data. Note that this is unlike
  `class_weights` in that `class_weights` weights the example
  depending on the value of its label, whereas `label_weights` depends
  only on the index of that label before flattening; therefore
  `label_weights` should not be used for multi-class data.
- **from_logits**: boolean indicating whether the predictions (`y_pred` in
  `update_state`) are probabilities or sigmoid logits. As a rule of thumb,
  when using a keras loss, the `from_logits` constructor argument of the
  loss should match the AUC `from_logits` constructor argument.

**Example**

```console
>>> m = keras.metrics.AUC(num_thresholds=3)
>>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
>>> # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
>>> # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
>>> # tp_rate = recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
>>> # auc = ((((1 + 0.5) / 2) * (1 - 0)) + (((0.5 + 0) / 2) * (0 - 0)))
>>> #     = 0.75
>>> m.result()
0.75
```

```console
>>> m.reset_state()
>>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
...                sample_weight=[1, 0, 0, 1])
>>> m.result()
1.0
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.AUC()])
model.compile(optimizer='sgd',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.AUC(from_logits=True)])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L254" >}}

### `Precision` class

```python
keras.metrics.Precision(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)
```

Computes the precision of the predictions with respect to the labels.

The metric creates two local variables, `true_positives` and
`false_positives` that are used to compute the precision. This value is
ultimately returned as `precision`, an idempotent operation that simply
divides `true_positives` by the sum of `true_positives` and
`false_positives`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `top_k` is set, we'll calculate precision as how often on average a class
among the top-k classes with the highest predicted values of a batch entry
is correct and can be found in the label for that entry.

If `class_id` is specified, we calculate precision by considering only the
entries in the batch for which `class_id` is above the threshold and/or in
the top-k highest predictions, and computing the fraction of them for which
`class_id` is indeed a correct label.

**Arguments**

- **thresholds**: (Optional) A float value, or a Python list/tuple of float
  threshold values in `[0, 1]`. A threshold is compared with
  prediction values to determine the truth value of predictions (i.e.,
  above the threshold is `True`, below is `False`). If used with a
  loss function that sets `from_logits=True` (i.e. no sigmoid applied
  to predictions), `thresholds` should be set to 0. One metric value
  is generated for each threshold value. If neither `thresholds` nor
  `top_k` are set, the default is to calculate precision with
  `thresholds=0.5`.
- **top_k**: (Optional) Unset by default. An int value specifying the top-k
  predictions to consider when calculating precision.
- **class_id**: (Optional) Integer class ID for which we want binary metrics.
  This must be in the half-open interval `[0, num_classes)`, where
  `num_classes` is the last dimension of predictions.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.Precision()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
>>> m.result()
0.6666667
```

```console
>>> m.reset_state()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result()
1.0
```

```console
>>> # With top_k=2, it will calculate precision over y_true[:2]
>>> # and y_pred[:2]
>>> m = keras.metrics.Precision(top_k=2)
>>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
>>> m.result()
0.0
```

```console
>>> # With top_k=4, it will calculate precision over y_true[:4]
>>> # and y_pred[:4]
>>> m = keras.metrics.Precision(top_k=4)
>>> m.update_state([0, 0, 1, 1], [1, 1, 1, 1])
>>> m.result()
0.5
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=[keras.metrics.Precision()])
```

Usage with a loss with `from_logits=True`:

```python
model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.Precision(thresholds=0)])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L414" >}}

### `Recall` class

```python
keras.metrics.Recall(
    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
)
```

Computes the recall of the predictions with respect to the labels.

This metric creates two local variables, `true_positives` and
`false_negatives`, that are used to compute the recall. This value is
ultimately returned as `recall`, an idempotent operation that simply divides
`true_positives` by the sum of `true_positives` and `false_negatives`.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `top_k` is set, recall will be computed as how often on average a class
among the labels of a batch entry is in the top-k predictions.

If `class_id` is specified, we calculate recall by considering only the
entries in the batch for which `class_id` is in the label, and computing the
fraction of them for which `class_id` is above the threshold and/or in the
top-k predictions.

**Arguments**

- **thresholds**: (Optional) A float value, or a Python list/tuple of float
  threshold values in `[0, 1]`. A threshold is compared with
  prediction values to determine the truth value of predictions (i.e.,
  above the threshold is `True`, below is `False`). If used with a
  loss function that sets `from_logits=True` (i.e. no sigmoid
  applied to predictions), `thresholds` should be set to 0.
  One metric value is generated for each threshold value.
  If neither `thresholds` nor `top_k` are set,
  the default is to calculate recall with `thresholds=0.5`.
- **top_k**: (Optional) Unset by default. An int value specifying the top-k
  predictions to consider when calculating recall.
- **class_id**: (Optional) Integer class ID for which we want binary metrics.
  This must be in the half-open interval `[0, num_classes)`, where
  `num_classes` is the last dimension of predictions.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.Recall()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
>>> m.result()
0.6666667
```

```console
>>> m.reset_state()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result()
1.0
```

Usage with `compile()` API:

```python
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=[keras.metrics.Recall()])
```

Usage with a loss with `from_logits=True`:

```python
model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.Recall(thresholds=0)])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L210" >}}

### `TruePositives` class

```python
keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
```

Calculates the number of true positives.

If `sample_weight` is given, calculates the sum of the weights of
true positives. This metric creates one local variable, `true_positives`
that is used to keep track of the number of true positives.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

**Arguments**

- **thresholds**: (Optional) Defaults to `0.5`. A float value, or a Python
  list/tuple of float threshold values in `[0, 1]`. A threshold is
  compared with prediction values to determine the truth value of
  predictions (i.e., above the threshold is `True`, below is `False`).
  If used with a loss function that sets `from_logits=True` (i.e. no
  sigmoid applied to predictions), `thresholds` should be set to 0.
  One metric value is generated for each threshold value.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.TruePositives()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
>>> m.result()
2.0
```

```console
>>> m.reset_state()
>>> m.update_state([0, 1, 1, 1], [1, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result()
1.0
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L166" >}}

### `TrueNegatives` class

```python
keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
```

Calculates the number of true negatives.

If `sample_weight` is given, calculates the sum of the weights of
true negatives. This metric creates one local variable, `accumulator`
that is used to keep track of the number of true negatives.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

**Arguments**

- **thresholds**: (Optional) Defaults to `0.5`. A float value, or a Python
  list/tuple of float threshold values in `[0, 1]`. A threshold is
  compared with prediction values to determine the truth value of
  predictions (i.e., above the threshold is `True`, below is `False`).
  If used with a loss function that sets `from_logits=True` (i.e. no
  sigmoid applied to predictions), `thresholds` should be set to 0.
  One metric value is generated for each threshold value.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.TrueNegatives()
>>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0])
>>> m.result()
2.0
```

```console
>>> m.reset_state()
>>> m.update_state([0, 1, 0, 0], [1, 1, 0, 0], sample_weight=[0, 0, 1, 0])
>>> m.result()
1.0
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L78" >}}

### `FalsePositives` class

```python
keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
```

Calculates the number of false positives.

If `sample_weight` is given, calculates the sum of the weights of
false positives. This metric creates one local variable, `accumulator`
that is used to keep track of the number of false positives.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

**Arguments**

- **thresholds**: (Optional) Defaults to `0.5`. A float value, or a Python
  list/tuple of float threshold values in `[0, 1]`. A threshold is
  compared with prediction values to determine the truth value of
  predictions (i.e., above the threshold is `True`, below is `False`).
  If used with a loss function that sets `from_logits=True` (i.e. no
  sigmoid applied to predictions), `thresholds` should be set to 0.
  One metric value is generated for each threshold value.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Examples**

```console
>>> m = keras.metrics.FalsePositives()
>>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1])
>>> m.result()
2.0
```

```console
>>> m.reset_state()
>>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1], sample_weight=[0, 0, 1, 0])
>>> m.result()
1.0
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L122" >}}

### `FalseNegatives` class

```python
keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)
```

Calculates the number of false negatives.

If `sample_weight` is given, calculates the sum of the weights of
false negatives. This metric creates one local variable, `accumulator`
that is used to keep track of the number of false negatives.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

**Arguments**

- **thresholds**: (Optional) Defaults to `0.5`. A float value, or a Python
  list/tuple of float threshold values in `[0, 1]`. A threshold is
  compared with prediction values to determine the truth value of
  predictions (i.e., above the threshold is `True`, below is `False`).
  If used with a loss function that sets `from_logits=True` (i.e. no
  sigmoid applied to predictions), `thresholds` should be set to 0.
  One metric value is generated for each threshold value.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.FalseNegatives()
>>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
>>> m.result()
2.0
```

```console
>>> m.reset_state()
>>> m.update_state([0, 1, 1, 1], [0, 1, 0, 0], sample_weight=[0, 0, 1, 0])
>>> m.result()
1.0
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L882" >}}

### `PrecisionAtRecall` class

```python
keras.metrics.PrecisionAtRecall(
    recall, num_thresholds=200, class_id=None, name=None, dtype=None
)
```

Computes best precision where recall is >= specified value.

This metric creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the precision at the given recall. The threshold for the given
recall value is computed and used to evaluate the corresponding precision.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `class_id` is specified, we calculate precision by considering only the
entries in the batch for which `class_id` is above the threshold
predictions, and computing the fraction of them for which `class_id` is
indeed a correct label.

**Arguments**

- **recall**: A scalar value in range `[0, 1]`.
- **num_thresholds**: (Optional) Defaults to 200. The number of thresholds to
  use for matching the given recall.
- **class_id**: (Optional) Integer class ID for which we want binary metrics.
  This must be in the half-open interval `[0, num_classes)`, where
  `num_classes` is the last dimension of predictions.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.PrecisionAtRecall(0.5)
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
>>> m.result()
0.5
```

```console
>>> m.reset_state()
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
...                sample_weight=[2, 2, 2, 1, 1])
>>> m.result()
0.33333333
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=[keras.metrics.PrecisionAtRecall(recall=0.8)])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L969" >}}

### `RecallAtPrecision` class

```python
keras.metrics.RecallAtPrecision(
    precision, num_thresholds=200, class_id=None, name=None, dtype=None
)
```

Computes best recall where precision is >= specified value.

For a given score-label-distribution the required precision might not
be achievable, in this case 0.0 is returned as recall.

This metric creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the recall at the given precision. The threshold for the given
precision value is computed and used to evaluate the corresponding recall.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `class_id` is specified, we calculate precision by considering only the
entries in the batch for which `class_id` is above the threshold
predictions, and computing the fraction of them for which `class_id` is
indeed a correct label.

**Arguments**

- **precision**: A scalar value in range `[0, 1]`.
- **num_thresholds**: (Optional) Defaults to 200. The number of thresholds
  to use for matching the given precision.
- **class_id**: (Optional) Integer class ID for which we want binary metrics.
  This must be in the half-open interval `[0, num_classes)`, where
  `num_classes` is the last dimension of predictions.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.RecallAtPrecision(0.8)
>>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
>>> m.result()
0.5
```

```console
>>> m.reset_state()
>>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
...                sample_weight=[1, 0, 0, 1])
>>> m.result()
1.0
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=[keras.metrics.RecallAtPrecision(precision=0.8)])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L674" >}}

### `SensitivityAtSpecificity` class

```python
keras.metrics.SensitivityAtSpecificity(
    specificity, num_thresholds=200, class_id=None, name=None, dtype=None
)
```

Computes best sensitivity where specificity is >= specified value.

`Sensitivity` measures the proportion of actual positives that are correctly
identified as such `(tp / (tp + fn))`.
`Specificity` measures the proportion of actual negatives that are correctly
identified as such `(tn / (tn + fp))`.

This metric creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the sensitivity at the given specificity. The threshold for the
given specificity value is computed and used to evaluate the corresponding
sensitivity.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `class_id` is specified, we calculate precision by considering only the
entries in the batch for which `class_id` is above the threshold
predictions, and computing the fraction of them for which `class_id` is
indeed a correct label.

For additional information about specificity and sensitivity, see
[the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

**Arguments**

- **specificity**: A scalar value in range `[0, 1]`.
- **num_thresholds**: (Optional) Defaults to 200. The number of thresholds to
  use for matching the given specificity.
- **class_id**: (Optional) Integer class ID for which we want binary metrics.
  This must be in the half-open interval `[0, num_classes)`, where
  `num_classes` is the last dimension of predictions.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.SensitivityAtSpecificity(0.5)
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
>>> m.result()
0.5
```

```console
>>> m.reset_state()
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
...                sample_weight=[1, 1, 2, 2, 1])
>>> m.result()
0.333333
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=[keras.metrics.SensitivityAtSpecificity()])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/confusion_metrics.py#L778" >}}

### `SpecificityAtSensitivity` class

```python
keras.metrics.SpecificityAtSensitivity(
    sensitivity, num_thresholds=200, class_id=None, name=None, dtype=None
)
```

Computes best specificity where sensitivity is >= specified value.

`Sensitivity` measures the proportion of actual positives that are correctly
identified as such `(tp / (tp + fn))`.
`Specificity` measures the proportion of actual negatives that are correctly
identified as such `(tn / (tn + fp))`.

This metric creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the specificity at the given sensitivity. The threshold for the
given sensitivity value is computed and used to evaluate the corresponding
specificity.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

If `class_id` is specified, we calculate precision by considering only the
entries in the batch for which `class_id` is above the threshold
predictions, and computing the fraction of them for which `class_id` is
indeed a correct label.

For additional information about specificity and sensitivity, see
[the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).

**Arguments**

- **sensitivity**: A scalar value in range `[0, 1]`.
- **num_thresholds**: (Optional) Defaults to 200. The number of thresholds to
  use for matching the given sensitivity.
- **class_id**: (Optional) Integer class ID for which we want binary metrics.
  This must be in the half-open interval `[0, num_classes)`, where
  `num_classes` is the last dimension of predictions.
- **name**: (Optional) string name of the metric instance.
- **dtype**: (Optional) data type of the metric result.

**Example**

```console
>>> m = keras.metrics.SpecificityAtSensitivity(0.5)
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8])
>>> m.result()
0.66666667
```

```console
>>> m.reset_state()
>>> m.update_state([0, 0, 0, 1, 1], [0, 0.3, 0.8, 0.3, 0.8],
...                sample_weight=[1, 1, 2, 2, 2])
>>> m.result()
0.5
```

Usage with `compile()` API:

```python
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=[keras.metrics.SpecificityAtSensitivity()])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/f_score_metrics.py#L250" >}}

### `F1Score` class

```python
keras.metrics.F1Score(average=None, threshold=None, name="f1_score", dtype=None)
```

Computes F-1 Score.

Formula:

```python
f1_score = 2 * (precision * recall) / (precision + recall)
```

This is the harmonic mean of precision and recall.
Its output range is `[0, 1]`. It works for both multi-class
and multi-label classification.

**Arguments**

- **average**: Type of averaging to be performed on data.
  Acceptable values are `None`, `"micro"`, `"macro"`
  and `"weighted"`. Defaults to `None`.
  If `None`, no averaging is performed and `result()` will return
  the score for each class.
  If `"micro"`, compute metrics globally by counting the total
  true positives, false negatives and false positives.
  If `"macro"`, compute metrics for each label,
  and return their unweighted mean.
  This does not take label imbalance into account.
  If `"weighted"`, compute metrics for each label,
  and return their average weighted by support
  (the number of true instances for each label).
  This alters `"macro"` to account for label imbalance.
  It can result in an score that is not between precision and recall.
- **threshold**: Elements of `y_pred` greater than `threshold` are
  converted to be 1, and the rest 0. If `threshold` is
  `None`, the argmax of `y_pred` is converted to 1, and the rest to 0.
- **name**: Optional. String name of the metric instance.
- **dtype**: Optional. Data type of the metric result.

**Returns**

- **F-1 Score**: float.

**Example**

```console
>>> metric = keras.metrics.F1Score(threshold=0.5)
>>> y_true = np.array([[1, 1, 1],
...                    [1, 0, 0],
...                    [1, 1, 0]], np.int32)
>>> y_pred = np.array([[0.2, 0.6, 0.7],
...                    [0.2, 0.6, 0.6],
...                    [0.6, 0.8, 0.0]], np.float32)
>>> metric.update_state(y_true, y_pred)
>>> result = metric.result()
array([0.5      , 0.8      , 0.6666667], dtype=float32)
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/metrics/f_score_metrics.py#L8" >}}

### `FBetaScore` class

```python
keras.metrics.FBetaScore(
    average=None, beta=1.0, threshold=None, name="fbeta_score", dtype=None
)
```

Computes F-Beta score.

Formula:

```python
b2 = beta ** 2
f_beta_score = (1 + b2) * (precision * recall) / (precision * b2 + recall)
```

This is the weighted harmonic mean of precision and recall.
Its output range is `[0, 1]`. It works for both multi-class
and multi-label classification.

**Arguments**

- **average**: Type of averaging to be performed across per-class results
  in the multi-class case.
  Acceptable values are `None`, `"micro"`, `"macro"` and
  `"weighted"`. Defaults to `None`.
  If `None`, no averaging is performed and `result()` will return
  the score for each class.
  If `"micro"`, compute metrics globally by counting the total
  true positives, false negatives and false positives.
  If `"macro"`, compute metrics for each label,
  and return their unweighted mean.
  This does not take label imbalance into account.
  If `"weighted"`, compute metrics for each label,
  and return their average weighted by support
  (the number of true instances for each label).
  This alters `"macro"` to account for label imbalance.
  It can result in an score that is not between precision and recall.
- **beta**: Determines the weight of given to recall
  in the harmonic mean between precision and recall (see pseudocode
  equation above). Defaults to `1`.
- **threshold**: Elements of `y_pred` greater than `threshold` are
  converted to be 1, and the rest 0. If `threshold` is
  `None`, the argmax of `y_pred` is converted to 1, and the rest to 0.
- **name**: Optional. String name of the metric instance.
- **dtype**: Optional. Data type of the metric result.

**Returns**

- **F-Beta Score**: float.

**Example**

```console
>>> metric = keras.metrics.FBetaScore(beta=2.0, threshold=0.5)
>>> y_true = np.array([[1, 1, 1],
...                    [1, 0, 0],
...                    [1, 1, 0]], np.int32)
>>> y_pred = np.array([[0.2, 0.6, 0.7],
...                    [0.2, 0.6, 0.6],
...                    [0.6, 0.8, 0.0]], np.float32)
>>> metric.update_state(y_true, y_pred)
>>> result = metric.result()
>>> result
[0.3846154 , 0.90909094, 0.8333334 ]
```
