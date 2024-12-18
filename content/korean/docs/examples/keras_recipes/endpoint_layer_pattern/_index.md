---
title: 엔드포인트 레이어 패턴
linkTitle: 엔드포인트 레이어 패턴
toc: true
weight: 7
type: docs
---

{{< keras/original checkedAt="2024-11-24" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2019/05/10  
**{{< t f_last_modified >}}** 2023/11/22  
**{{< t f_description >}}** Demonstration of the "endpoint layer" pattern (layer that handles loss management).

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/endpoint_layer_pattern.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/endpoint_layer_pattern.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Setup {#setup}

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
import numpy as np
```

## Usage of endpoint layers in the Functional API {#usage-of-endpoint-layers-in-the-functional-api}

An "endpoint layer" has access to the model's targets, and creates arbitrary losses in `call()` using `self.add_loss()` and `Metric.update_state()`. This enables you to define losses and metrics that don't match the usual signature `fn(y_true, y_pred, sample_weight=None)`.

Note that you could have separate metrics for training and eval with this pattern.

```python
class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_metric = keras.metrics.BinaryAccuracy(name="accuracy")

    def call(self, logits, targets=None, sample_weight=None):
        if targets is not None:
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            loss = self.loss_fn(targets, logits, sample_weight)
            self.add_loss(loss)

            # Log the accuracy as a metric (we could log arbitrary metrics,
            # including different metrics for training and inference.)
            self.accuracy_metric.update_state(targets, logits, sample_weight)

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)


inputs = keras.Input((764,), name="inputs")
logits = keras.layers.Dense(1)(inputs)
targets = keras.Input((1,), name="targets")
sample_weight = keras.Input((1,), name="sample_weight")
preds = LogisticEndpoint()(logits, targets, sample_weight)
model = keras.Model([inputs, targets, sample_weight], preds)

data = {
    "inputs": np.random.random((1000, 764)),
    "targets": np.random.random((1000, 1)),
    "sample_weight": np.random.random((1000, 1)),
}

model.compile(keras.optimizers.Adam(1e-3))
model.fit(data, epochs=2)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/2
 27/32 ━━━━━━━━━━━━━━━━[37m━━━━  0s 2ms/step - loss: 0.3664

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1700705222.380735 3351467 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

 32/32 ━━━━━━━━━━━━━━━━━━━━ 2s 31ms/step - loss: 0.3663
Epoch 2/2
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.3627

<keras.src.callbacks.history.History at 0x7f13401b1e10>
```

{{% /details %}}

## Exporting an inference-only model {#exporting-an-inference-only-model}

Simply don't include `targets` in the model. The weights stay the same.

```python
inputs = keras.Input((764,), name="inputs")
logits = keras.layers.Dense(1)(inputs)
preds = LogisticEndpoint()(logits, targets=None, sample_weight=None)
inference_model = keras.Model(inputs, preds)

inference_model.set_weights(model.get_weights())

preds = inference_model.predict(np.random.random((1000, 764)))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step
```

{{% /details %}}

## Usage of loss endpoint layers in subclassed models {#usage-of-loss-endpoint-layers-in-subclassed-models}

```python
class LogReg(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(1)
        self.logistic_endpoint = LogisticEndpoint()

    def call(self, inputs):
        # Note that all inputs should be in the first argument
        # since we want to be able to call `model.fit(inputs)`.
        logits = self.dense(inputs["inputs"])
        preds = self.logistic_endpoint(
            logits=logits,
            targets=inputs["targets"],
            sample_weight=inputs["sample_weight"],
        )
        return preds


model = LogReg()
data = {
    "inputs": np.random.random((1000, 764)),
    "targets": np.random.random((1000, 1)),
    "sample_weight": np.random.random((1000, 1)),
}

model.compile(keras.optimizers.Adam(1e-3))
model.fit(data, epochs=2)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/2
 32/32 ━━━━━━━━━━━━━━━━━━━━ 1s 9ms/step - loss: 0.3529
Epoch 2/2
 32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.3509

<keras.src.callbacks.history.History at 0x7f132c1d1450>
```

{{% /details %}}
