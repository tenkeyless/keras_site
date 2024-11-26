---
title: The base HyperModel class
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/hypermodel.py#L20" >}}

### `HyperModel` class

```python
keras_tuner.HyperModel(name=None, tunable=True)
```

Defines a search space of models.

A search space is a collection of models. The `build` function will build
one of the models from the space using the given `HyperParameters` object.

Users should subclass the `HyperModel` class to define their search spaces
by overriding `build()`, which creates and returns the Keras model.
Optionally, you may also override `fit()` to customize the training process
of the model.

**Examples**

In `build()`, you can create the model using the hyperparameters.

```python
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Dense(
            hp.Choice('units', [8, 16, 32]),
            activation='relu'))
        model.add(keras.layers.Dense(1, activation='relu'))
        model.compile(loss='mse')
        return model
```

When overriding `HyperModel.fit()`, if you use `model.fit()` to train your
model, which returns the training history, you can return it directly. You
may use `hp` to specify any hyperparameters to tune.

```python
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        ...
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            epochs=hp.Int("epochs", 5, 20),
            **kwargs)
```

If you have a customized training process, you can return the objective
value as a float.

If you want to keep track of more metrics, you can return a dictionary of
the metrics to track.

```python
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        ...
    def fit(self, hp, model, *args, **kwargs):
        ...
        return {
            "loss": loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }
```

**Arguments**

- **name**: Optional string, the name of this HyperModel.
- **tunable**: Boolean, whether the hyperparameters defined in this
  hypermodel should be added to search space. If `False`, either the
  search space for these parameters must be defined in advance, or
  the default values will be used. Defaults to True.

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/engine/hypermodel.py#L104" >}}

### `build` method

```python
HyperModel.build(hp)
```

Builds a model.

**Arguments**

- **hp**: A `HyperParameters` instance.

**Returns**

A model instance.
