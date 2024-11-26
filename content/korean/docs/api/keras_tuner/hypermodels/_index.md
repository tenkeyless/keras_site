---
title: KerasTuner HyperModels
linkTitle: HyperModels
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

The `HyperModel` base class makes the search space better encapsulated for
sharing and reuse. A `HyperModel` subclass only needs to implement a
`build(self, hp)` method, which creates a [`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}) using the `hp` argument
to define the hyperparameters and returns the model instance.
A simple code example is shown as follows.

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

You can pass a `HyperModel` instance to the `Tuner` as the search space.

```python
tuner = kt.RandomSearch(
    MyHyperModel(),
    objective='val_loss',
    max_trials=5)
```

There are also some built-in `HyperModel` subclasses (e.g. `HyperResNet`,
`HyperXception`) for the users to directly use so that the users don't need to
write their own search spaces.

```python
tuner = kt.RandomSearch(
    HyperResNet(input_shape=(28, 28, 1), classes=10),
    objective='val_loss',
    max_trials=5)
```

### [The base HyperModel class]({{< relref "/docs/api/keras_tuner/hypermodels/base_hypermodel/" >}})

- [HyperModel class]({{< relref "/docs/api/keras_tuner/hypermodels/base_hypermodel/#hypermodel-class" >}})
- [build method]({{< relref "/docs/api/keras_tuner/hypermodels/base_hypermodel/#build-method" >}})

### [HyperEfficientNet]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_efficientnet/" >}})

- [HyperEfficientNet class]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_efficientnet/#hyperefficientnet-class" >}})

### [HyperImageAugment]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_image_augment/" >}})

- [HyperImageAugment class]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_image_augment/#hyperimageaugment-class" >}})

### [HyperResNet]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_resnet/" >}})

- [HyperResNet class]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_resnet/#hyperresnet-class" >}})

### [HyperXception]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_xception/" >}})

- [HyperXception class]({{< relref "/docs/api/keras_tuner/hypermodels/hyper_xception/#hyperxception-class" >}})
