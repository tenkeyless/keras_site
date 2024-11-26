---
title: Sklearn Tuner
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/tuners/sklearn_tuner.py#L51" >}}

### `SklearnTuner` class

```python
keras_tuner.SklearnTuner(
    oracle, hypermodel, scoring=None, metrics=None, cv=None, **kwargs
)
```

Tuner for Scikit-learn Models.

Performs cross-validated hyperparameter search for Scikit-learn models.

**Examples**

```python
import keras_tuner
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
def build_model(hp):
  model_type = hp.Choice('model_type', ['random_forest', 'ridge'])
  if model_type == 'random_forest':
    model = ensemble.RandomForestClassifier(
        n_estimators=hp.Int('n_estimators', 10, 50, step=10),
        max_depth=hp.Int('max_depth', 3, 10))
  else:
    model = linear_model.RidgeClassifier(
        alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))
  return model
tuner = keras_tuner.tuners.SklearnTuner(
    oracle=keras_tuner.oracles.BayesianOptimizationOracle(
        objective=keras_tuner.Objective('score', 'max'),
        max_trials=10),
    hypermodel=build_model,
    scoring=metrics.make_scorer(metrics.accuracy_score),
    cv=model_selection.StratifiedKFold(5),
    directory='.',
    project_name='my_project')
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2)
tuner.search(X_train, y_train)
best_model = tuner.get_best_models(num_models=1)[0]
```

**Arguments**

- **oracle**: A [`keras_tuner.Oracle`]({{< relref "/docs/api/keras_tuner/oracles/base_oracle#oracle-class" >}}) instance. Note that for this `Tuner`,
  the `objective` for the `Oracle` should always be set to
  `Objective('score', direction='max')`. Also, `Oracle`s that exploit
  Neural-Network-specific training (e.g. `Hyperband`) should not be
  used with this `Tuner`.
- **hypermodel**: A `HyperModel` instance (or callable that takes
  hyperparameters and returns a Model instance).
- **scoring**: An sklearn `scoring` function. For more information, see
  `sklearn.metrics.make_scorer`. If not provided, the Model's default
  scoring will be used via `model.score`. Note that if you are
  searching across different Model families, the default scoring for
  these Models will often be different. In this case you should
  supply `scoring` here in order to make sure your Models are being
  scored on the same metric.
- **metrics**: Additional `sklearn.metrics` functions to monitor during
  search. Note that these metrics do not affect the search process.
- **cv**: An `sklearn.model_selection` Splitter class. Used to
  determine how samples are split up into groups for
  cross-validation.
- **\*\*kwargs**: Keyword arguments relevant to all `Tuner` subclasses. Please
  see the docstring for `Tuner`.
