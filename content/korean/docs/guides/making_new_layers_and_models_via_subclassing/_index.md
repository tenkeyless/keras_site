---
title: 서브클래싱을 통해 새로운 레이어와 모델 만들기
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2019/03/01  
**{{< t f_last_modified >}}** 2023/06/25  
**{{< t f_description >}}** `Layer`와 `Model` 객체를 처음부터 작성하는 방법에 대한 완벽 가이드.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/making_new_layers_and_models_via_subclassing.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/making_new_layers_and_models_via_subclassing.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

이 가이드에서는 자체 서브클래싱된 레이어와 모델을 빌드하는 데 필요한 모든 내용을 다룹니다.
특히, 다음 기능에 대해 알아봅니다.

- `Layer` 클래스
- `add_weight()` 메서드
- 트레이닝 가능한 가중치와 트레이닝 불가능한 가중치
- `build()` 메서드
- 레이어를 모든 백엔드에서 사용할 수 있는지 확인
- `add_loss()` 메서드
- `call()`의 `training` 인수
- `call()`의 `mask` 인수
- 레이어를 직렬화할 수 있는지 확인

자세히 알아보겠습니다.

## 셋업 {#setup}

```python
import numpy as np
import keras
from keras import ops
from keras import layers
```

## `Layer` 클래스: 상태(가중치)와 일부 계산의 조합 {#the-layer-class-the-combination-of-state-weights-and-some-computation}

Keras의 중심 추상화 중 하나는 `Layer` 클래스입니다.
레이어는 상태(레이어의 "가중치")와 입력에서 출력으로의 변환(레이어의 포워드 패스인 "호출")을 모두 캡슐화합니다.

다음은 밀집 연결(densely-connected) 레이어입니다.
여기에는 두 개의 상태 변수가 있습니다. 변수 `w`와 `b`입니다.

```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b
```

(Python 함수처럼) 텐서 입력에 대해 호출하여 레이어를 사용합니다.

```python
x = ops.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
[[ 0.085416   -0.06821361 -0.00741937 -0.03429271]
 [ 0.085416   -0.06821361 -0.00741937 -0.03429271]]
```

{{% /details %}}

가중치 `w`와 `b`는 레이어 속성으로 설정되면, 레이어에서 자동으로 추적됩니다.

```python
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```

## 레이어는 트레이닝 불가능한 가중치를 가질 수 있습니다. {#layers-can-have-non-trainable-weights}

트레이닝 가능한 가중치 외에도, 레이어에 트레이닝 불가능한 가중치를 추가할 수 있습니다.
이러한 가중치는, 레이어를 트레이닝할 때, 역전파 중에 고려되지 않아야 합니다.

트레이닝 불가능한 가중치를 추가하고 사용하는 방법은 다음과 같습니다.

```python
class ComputeSum(keras.layers.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.total = self.add_weight(
            initializer="zeros", shape=(input_dim,), trainable=False
        )

    def call(self, inputs):
        self.total.assign_add(ops.sum(inputs, axis=0))
        return self.total


x = ops.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
[2. 2.]
[4. 4.]
```

{{% /details %}}

이는 `layer.weights`의 일부이지만, 트레이닝 불가능한 가중치로 분류됩니다.

```python
print("weights:", len(my_sum.weights))
print("non-trainable weights:", len(my_sum.non_trainable_weights))

# 트레이닝 가능한 가중치에 포함되지 않습니다.
print("trainable_weights:", my_sum.trainable_weights)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
weights: 1
non-trainable weights: 1
trainable_weights: []
```

{{% /details %}}

## 모범 사례: 입력 모양이 알려질 때까지, 가중치 생성을 연기합니다. {#best-practice-deferring-weight-creation-until-the-shape-of-the-inputs-is-known}

우리의 `Linear` 레이어는, `__init__()`에서 가중치 `w`와 `b`의 모양을 계산하는 데 사용된,
`input_dim` 인수를 사용했습니다.

```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b
```

많은 경우, 입력 크기를 미리 알 수 없으며,
레이어를 인스턴스화한 얼마간 이후, 해당 값이 알려질 때,
지연하여(lazily) 가중치를 생성하고 싶을 것입니다.

Keras API에서, 레이어의 `build(self, inputs_shape)` 메서드에서 레이어 가중치를 생성하는 것이 좋습니다.
다음과 같이 합니다.

```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b
```

레이어의 `__call__()` 메서드는 처음 호출될 때 자동으로 빌드를 실행합니다.
이제 지연되고(lazy), 사용하기 쉬운 레이어가 생겼습니다.

```python
# 인스턴스화 시, 이것이 어떤 입력에 대해 호출될지 알 수 없습니다.
linear_layer = Linear(32)

# 레이어의 가중치는 레이어가 처음 호출될 때 동적으로 생성됩니다.
y = linear_layer(x)
```

위에 표시된 것처럼 `build()`를 별도로 구현하면,
가중치를 한 번만 생성하는 것과,
모든 호출에서 가중치를 사용하는 것을 깔끔하게 분리할 수 있습니다.

## 레이어는 재귀적으로 구성 가능합니다. {#layers-are-recursively-composable}

Layer 인스턴스를 다른 Layer의 속성으로 할당하면,
외부 레이어는 내부 레이어에서 생성된 가중치를 추적하기 시작합니다.

우리는 `__init__()` 메서드에서 이러한 하위 레이어를 생성하고,
첫 번째 `__call__()`에서 그들의 가중치를 빌드하도록 트리거하는 것을 권장합니다.

```python
class MLPBlock(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = keras.activations.relu(x)
        x = self.linear_2(x)
        x = keras.activations.relu(x)
        return self.linear_3(x)


mlp = MLPBlock()
y = mlp(ops.ones(shape=(3, 64)))  # `mlp`에 대한 첫 번째 호출은 가중치를 생성합니다.
print("weights:", len(mlp.weights))
print("trainable weights:", len(mlp.trainable_weights))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
weights: 6
trainable weights: 6
```

{{% /details %}}

## 백엔드에 독립적인 레이어와 백엔드에 특화된 레이어 {#backend-agnostic-layers-and-backend-specific-layers}

레이어가 `keras.ops` 네임스페이스(또는 `keras.activations`, `keras.random` 또는 `keras.layers`와 같은 그 외 Keras 네임스페이스)의 API만 사용하는 한,
TensorFlow, JAX 또는 PyTorch와 같은 모든 백엔드에서 사용할 수 있습니다.

이 가이드에서 지금까지 본 모든 레이어는 모든 Keras 백엔드에서 작동합니다.

`keras.ops` 네임스페이스는 다음에 대한 액세스를 제공합니다.

- NumPy API, 예: `ops.matmul`, `ops.sum`, `ops.reshape`, `ops.stack` 등
- `ops.softmax`, `ops.conv`, `ops.binary_crossentropy`, `ops.relu` 등과 같은 신경망 전용 API

레이어에서 백엔드 네이티브 API(예: [`tf.nn`](https://www.tensorflow.org/api_docs/python/tf/nn) 함수)를 사용할 수도 있지만,
이렇게 하면 레이어는 해당 백엔드에서만 사용할 수 있습니다.
예를 들어, `jax.numpy`를 사용하여 다음과 같은 JAX 전용 레이어를 작성할 수 있습니다.

```python
import jax

class Linear(keras.layers.Layer):
    ...

    def call(self, inputs):
        return jax.numpy.matmul(inputs, self.w) + self.b
```

이것은 동등한 TensorFlow 전용 레이어 입니다.

```python
import tensorflow as tf

class Linear(keras.layers.Layer):
    ...

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

그리고 이것은 동등한 PyTorch 전용 레이어 입니다:

```python
import torch

class Linear(keras.layers.Layer):
    ...

    def call(self, inputs):
        return torch.matmul(inputs, self.w) + self.b
```

크로스 백엔드 호환성은 매우 유용한 속성이므로,
Keras API만 활용하여 레이어를 항상 백엔드에 독립적으로 만드는 것이 좋습니다.

## `add_loss()` 메서드 {#the-addloss-method}

레이어의 `call()` 메서드를 작성할 때,
나중에 트레이닝 루프를 작성할 때, 사용하고 싶은 손실 텐서를 만들 수 있습니다.
`self.add_loss(value)`를 호출하면 가능합니다.

```python
# 활동 정규화 손실(activity regularization loss)을 생성하는 레이어
class ActivityRegularizationLayer(keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super().__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * ops.mean(inputs))
        return inputs
```

이러한 손실(내부 레이어에서 생성된 손실 포함)은 `layer.losses`를 통해 검색할 수 있습니다.
이 속성은 최상위 레이어에 대한 모든 `__call__()` 시작 시 재설정되므로,
`layer.losses`는 항상 마지막 전방 패스(forward pass) 중에 생성된 손실 값을 포함합니다.

```python
class OuterLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)

    def call(self, inputs):
        return self.activity_reg(inputs)


layer = OuterLayer()
assert len(layer.losses) == 0  # 레이어가 호출된 적이 없으므로 아직 손실이 없습니다.

_ = layer(ops.zeros((1, 1)))
assert len(layer.losses) == 1  # 우리는 하나의 손실 값을 생성했습니다.

# `layer.losses`는 각 `__call__` 시작 시 재설정됩니다.
_ = layer(ops.zeros((1, 1)))
assert len(layer.losses) == 1  # 이는 위의 호출 중에 발생한 손실입니다.
```

또한, `loss` 속성에는 내부 레이어의 가중치에 대해 생성된 정규화 손실도 포함됩니다.

```python
class OuterLayerWithKernelRegularizer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(
            32, kernel_regularizer=keras.regularizers.l2(1e-3)
        )

    def call(self, inputs):
        return self.dense(inputs)


layer = OuterLayerWithKernelRegularizer()
_ = layer(ops.zeros((1, 1)))

# 이는 위의 `kernel_regularizer`에 의해 생성된,
# `1e-3 * sum(layer.dense.kernel ** 2)`입니다.
print(layer.losses)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
[Array(0.00217911, dtype=float32)]
```

{{% /details %}}

이러한 손실은 커스텀 트레이닝 루프를 작성할 때 고려해야 합니다.

또한 `fit()`와 원활하게 작동합니다. (있는 경우, 자동으로 합산되어 메인 손실에 추가됩니다.):

```python
inputs = keras.Input(shape=(3,))
outputs = ActivityRegularizationLayer()(inputs)
model = keras.Model(inputs, outputs)

# `compile`에 손실이 전달된 손실이 있으면, 정규화 손실에 합산되어 추가됩니다.
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# 모델에는 이미 최소화해야 할 손실이 있으므로,
# 전방 전달(forward pass) 중 `add_loss` 호출을 통해,
# `compile`에서 아무런 손실도 전달하지 않는 것도 가능합니다!
model.compile(optimizer="adam")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step - loss: 0.2650
 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - loss: 0.0050

<keras.src.callbacks.history.History at 0x146f71960>
```

{{% /details %}}

## 선택적으로 레이어에서 직렬화를 활성화할 수 있습니다. {#you-can-optionally-enable-serialization-on-your-layers}

커스텀 레이어를 [함수형 모델]({{< relref "/docs/guides/functional_api" >}})의 일부로 직렬화할 필요가 있는 경우,
선택적으로 `get_config()` 메서드를 구현할 수 있습니다.

```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


# 이제 config에서 레이어를 다시 생성할 수 있습니다.
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
{'units': 64}
```

{{% /details %}}

베이스 `Layer` 클래스의 `__init__()` 메서드는 일부 키워드 인수, 특히 `name`과 `dtype`을 취합니다.
이러한 인수를 `__init__()`에서 부모 클래스에 전달하고, 레이어 config에 포함하는 것이 좋습니다.

```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
{'name': 'linear_7', 'trainable': True, 'dtype': 'float32', 'units': 64}
```

{{% /details %}}

레이어를 config에서 역직렬화할 때 더 많은 유연성이 필요한 경우,
`from_config()` 클래스 메서드를 재정의할 수도 있습니다.
이것은 `from_config()`의 베이스 구현입니다.

```python
def from_config(cls, config):
    return cls(**config)
```

직렬화 및 저장에 대해 자세히 알아보려면,
{{< titledRelref "/docs/guides/serialization_and_saving" >}} 가이드를 참조하세요.

## `call()` 메서드의 특권(Privileged) `training` 인수 {#privileged-training-argument-in-the-call-method}

일부 레이어, 특히 `BatchNormalization` 레이어와 `Dropout` 레이어는,
트레이닝 및 추론 중에 서로 다른 동작을 합니다.
이러한 레이어의 경우, `call()` 메서드에서 `training`(boolean) 인수를 노출하는 것이 표준 관행입니다.

`call()`에서 이 인수를 노출하면, 빌트인 트레이닝 및 평가 루프(예: `fit()`)가,
트레이닝 및 추론에서 레이어를 올바르게 사용할 수 있습니다.

```python
class CustomDropout(keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs, training=None):
        if training:
            return keras.random.dropout(
                inputs, rate=self.rate, seed=self.seed_generator
            )
        return inputs
```

## `call()` 메서드의 특권 `mask` 인수 {#privileged-mask-argument-in-the-call-method}

`call()`에서 지원하는 다른 특권 인수는 `mask` 인수입니다.

모든 Keras RNN 레이어에서 찾을 수 있습니다.
마스크는 boolean 텐서(입력의 타임스텝당 하나의 boolean 값)로,
시계열 데이터를 처리할 때 특정 입력 타임스텝을 건너뛰는 데 사용됩니다.

Keras는, 이전(prior) 레이어에서 마스크가 생성될 때,
이를 지원하는 레이어에 대해 올바른 `mask` 인수를 `__call__()`에 자동으로 전달합니다.
마스크를 생성하는 레이어는 `mask_zero=True`로 구성된 `Embedding` 레이어와 `Masking` 레이어입니다.

## `Model` 클래스 {#the-model-class}

일반적으로, `Layer` 클래스를 사용하여 내부 계산 블록을 정의하고,
`Model` 클래스를 사용하여 외부 모델(트레이닝할 객체)을 정의합니다.

예를 들어, ResNet50 모델에서는, `Layer`를 하위 클래스화하는 여러 ResNet 블록과,
전체 ResNet50 네트워크를 포괄하는 단일 `Model`이 있습니다.

`Model` 클래스는 `Layer`와 동일한 API를 사용하지만, 다음과 같은 차이점이 있습니다.

- 빌트인 트레이닝, 평가 및 예측 루프(`model.fit()`, `model.evaluate()`, `model.predict()`)를 노출합니다.
- `model.layers` 속성을 통해, 내부 레이어 리스트를 노출합니다.
- 저장 및 직렬화 API(`save()`, `save_weights()`...)를 노출합니다.

실제로, `Layer` 클래스는 문헌에서
"레이어"(예: "컨볼루션 레이어" 또는 "recurrent 레이어") 또는
"블록"(예: "ResNet 블록" 또는 "Inception 블록")이라고 하는 것과 일치합니다.

한편, `Model` 클래스는 문헌에서 "모델"(예: "딥러닝 모델") 또는 "네트워크"(예: "딥 신경망(신경 네트워크)")라고 하는 것에 해당합니다.

따라서, "`Layer` 클래스나 `Model` 클래스를 사용해야 할까?"라는 것이 궁금하다면 스스로에게 물어보세요.
`fit()`를 호출해야 할까요? `save()`를 호출해야 할까요? 그렇다면 `Model`을 사용하세요.
그렇지 않다면(클래스가 더 큰 시스템의 블록이거나 직접 트레이닝 및 저장 코드를 작성하고 있기 때문) `Layer`를 사용하세요.

예를 들어, 위의 mini-resnet 예제를 가져와, `fit()`로 트레이닝하고,
`save_weights()`로 저장할 수 있는 `Model`을 빌드할 수 있습니다.

```python
class ResNet(keras.Model):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)


resnet = ResNet()
dataset = ...
resnet.fit(dataset, epochs=10)
resnet.save(filepath.keras)
```

## 모두 합치기: 엔드투엔드 예시 {#putting-it-all-together-an-end-to-end-example}

지금까지 배운 내용은 다음과 같습니다.

- `Layer`는 상태(`__init__()` 또는 `build()`에서 생성)와 일부 계산(`call()`에서 정의)을 캡슐화합니다.
- 레이어는 재귀적으로 중첩되어, 새롭고 더 큰 계산 블록을 만들 수 있습니다.
- 레이어는 Keras API만 사용하는 한 백엔드에 구애받지 않습니다.
  백엔드 네이티브 API(예: `jax.numpy`, `torch.nn` 또는 [`tf.nn`](https://www.tensorflow.org/api_docs/python/tf/nn))를 사용할 수 있지만, 그러면 레이어는 해당 특정 백엔드에서만 사용할 수 있습니다.
- 레이어는 `add_loss()`를 통해 손실(일반적으로 정규화 손실)을 생성하고 추적할 수 있습니다.
- 외부 컨테이너, 즉 트레이닝하려는 것은 `Model`입니다.
  `Model`은 `Layer`와 같지만 트레이닝 및 직렬화 유틸리티가 추가되었습니다.

이 모든 것을 엔드투엔드 예제로 모아 보겠습니다. 
백엔드에 독립적인 방식으로 Variational AutoEncoder(VAE)를 구현하여, 
TensorFlow, JAX, PyTorch에서 동일하게 실행되도록 하겠습니다. 
MNIST 숫자에 대해 트레이닝 할 것입니다.

VAE는 `Model`의 서브클래스이며, 
`Layer`를 서브클래스하는 레이어의 중첩된 구성으로 구축됩니다. 
정규화 손실(KL 발산)이 특징입니다.

```python
class Sampling(layers.Layer):
    """(z_mean, z_log_var)를 사용하여 숫자를 인코딩하는 벡터 z를 샘플링합니다."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """MNIST 숫자를 삼중항(z_mean, z_log_var, z)으로 매핑합니다."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """인코딩된 숫자 벡터 z를 다시 읽을 수 있는 숫자로 변환합니다."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(keras.Model):
    """인코더와 디코더를 엔드투엔드 모델로 결합하여 트레이닝을 수행합니다."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # KL 발산 정규화 손실을 추가합니다.
        kl_loss = -0.5 * ops.mean(
            z_log_var - ops.square(z_mean) - ops.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed
```

`fit()` API를 사용하여 MNIST에 대해 트레이닝해 보겠습니다.

```python
(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255

original_dim = 784
vae = VariationalAutoEncoder(784, 64, 32)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=keras.losses.MeanSquaredError())

vae.fit(x_train, x_train, epochs=2, batch_size=64)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/2
 938/938 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - loss: 0.0942
Epoch 2/2
 938/938 ━━━━━━━━━━━━━━━━━━━━ 1s 859us/step - loss: 0.0677

<keras.src.callbacks.history.History at 0x146fe62f0>
```

{{% /details %}}
