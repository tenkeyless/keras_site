---
title: JAX로 멀티 GPU 분산 트레이닝하기
linkTitle: JAX 분산 트레이닝
toc: true
weight: 15
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2023/07/11  
**{{< t f_last_modified >}}** 2023/07/11  
**{{< t f_description >}}** JAX를 사용한 Keras 모델의 멀티 GPU/TPU 트레이닝 가이드.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/distributed_training_with_jax.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/distributed_training_with_jax.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

일반적으로 여러 장치에 연산을 분산시키는 방법에는 두 가지가 있습니다:

- **데이터 병렬 처리**
  - **데이터 병렬 처리**에서는 하나의 모델이 여러 장치나 여러 머신에 복제됩니다.
  - 각 장치는 서로 다른 배치의 데이터를 처리한 후, 결과를 병합합니다.
  - 이 설정에는 다양한 변형이 있으며, 서로 다른 모델 복제본이 결과를 병합하는 방식이나,
    각 배치마다 동기화되는지 여부 등에 차이가 있습니다.
- **모델 병렬 처리**
  - **모델 병렬 처리**에서는 하나의 모델의 다른 부분이 서로 다른 장치에서 실행되어, 하나의 데이터 배치를 함께 처리합니다.
  - 이는 여러 가지 브랜치를 특징으로 하는 자연스럽게 병렬화된 아키텍처를 가진 모델에 가장 적합합니다.

이 가이드는 데이터 병렬 처리, 특히 **동기식 데이터 병렬 처리**에 중점을 둡니다.
여기서 모델의 서로 다른 복제본은 각 배치를 처리한 후 동기화됩니다.
동기화는 모델의 수렴 동작을 단일 장치에서의 트레이닝과 동일하게 유지시킵니다.

구체적으로, 이 가이드는 최소한의 코드 변경으로 `jax.sharding` API를 사용하여,
Keras 모델을 여러 GPU 또는 TPU(일반적으로 2개에서 16개)를 사용하여, 단일 머신에서 트레이닝하는 방법을 설명합니다.
(단일 호스트, 다중 장치 트레이닝) 이는 연구자들과 소규모 산업 워크플로우에서 가장 흔한 설정입니다.

## 셋업 {#setup}

먼저 트레이닝할 모델을 생성하는 함수와,
트레이닝에 사용할 데이터셋을 생성하는 함수를 정의해봅시다. (이 경우 MNIST 데이터셋을 사용)

```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import numpy as np
import tensorflow as tf
import keras

from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def get_model():
    # 배치 정규화 및 드롭아웃을 포함한 간단한 컨볼루션 신경망을 만듭니다.
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(filters=12, kernel_size=3, padding="same", use_bias=False)(
        x
    )
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=24,
        kernel_size=6,
        use_bias=False,
        strides=2,
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=6,
        padding="same",
        strides=2,
        name="large_k",
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model


def get_datasets():
    # 데이터를 로드하고, 트레이닝 및 테스트 세트로 나눕니다.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 이미지를 [0, 1] 범위로 스케일링합니다.
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # 이미지가 (28, 28, 1) 형태를 갖추도록 만듭니다.
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # TF 데이터셋 생성
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    eval_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_data, eval_data
```

## 단일 호스트, 다중 장치 동기화 트레이닝 {#single-host-multi-device-synchronous-training}

이 설정에서는, 하나의 머신에 여러 개의 GPU 또는 TPU가 있습니다. (일반적으로 2~16개)
각 장치에서 모델의 복사본(**레플리카**)이 실행됩니다.
간단히 설명하기 위해, 다음 내용에서는 8개의 GPU를 사용하는 것으로 가정하겠습니다. 이는 일반성을 잃지 않습니다.​

**작동 방식**

트레이닝의 각 단계에서:

- **글로벌 배치**라고 불리는, 현재 배치 데이터가 8개의 서로 다른 **로컬 배치**로 나뉩니다.
  예를 들어, 글로벌 배치가 512개의 샘플을 포함하면, 각 로컬 배치는 64개의 샘플을 가집니다.
- 8개의 레플리카는 각각 로컬 배치를 독립적으로 처리합니다:
  순방향 전파 후 역전파를 수행하고, 로컬 배치에서 모델의 손실에 따른 가중치 그래디언트를 출력합니다.
- 로컬 그래디언트에서 발생한 가중치 업데이트는 8개의 레플리카 전반에서 효율적으로 병합됩니다.
  이는 각 단계가 끝날 때마다 이루어지므로, 레플리카는 항상 동기 상태를 유지합니다.

실제로, 모델 레플리카의 가중치를 동기화하는 과정은 각 개별 가중치 변수 레벨에서 처리됩니다.
이는 `jax.sharding.NamedSharding`을 사용하여, 변수들을 복제하는 방식으로 이루어집니다.

**사용 방법**

Keras 모델로 단일 호스트, 다중 장치 동기 트레이닝을 수행하려면,
`jax.sharding` 기능을 사용합니다. 사용 방법은 다음과 같습니다:

- 먼저 `mesh_utils.create_device_mesh`를 사용해, 장치 메쉬를 생성합니다.
- `jax.sharding.Mesh`, `jax.sharding.NamedSharding` 및 `jax.sharding.PartitionSpec`을 사용하여,
  JAX 배열을 어떻게 분할할지 정의합니다.
  - 모델과 옵티마이저 변수를 모든 장치에 복제하려면, 축이 없는 사양(a spec with no axis)을 사용합니다.
  - 데이터를 장치 간에 샤딩하려면, 배치 차원을 따라 분할하는 사양을 사용합니다.
- `jax.device_put`을 사용해 모델과 옵티마이저 변수를 장치 전반에 복제합니다.
  이는 처음에 한 번만 수행됩니다.
- 트레이닝 루프에서는 각 배치를 처리할 때,
  `jax.device_put`을 사용해 배치를 장치 전반에 분할한 후 트레이닝 단계를 호출합니다.

다음은 각 단계를 유틸리티 함수로 분할한 흐름입니다:

```python
# 설정
num_epochs = 2
batch_size = 64

train_data, eval_data = get_datasets()
train_data = train_data.batch(batch_size, drop_remainder=True)

model = get_model()
optimizer = keras.optimizers.Adam(1e-3)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 모든 상태를 .build()로 초기화
(one_batch, one_batch_labels) = next(iter(train_data))
model.build(one_batch)
optimizer.build(model.trainable_variables)


# 이 함수는 미분될 손실 함수입니다.
# Keras는 순수한 함수형 순방향 전파를 제공합니다: model.stateless_call
def compute_loss(trainable_variables, non_trainable_variables, x, y):
    y_pred, updated_non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x, training=True
    )
    loss_value = loss(y, y_pred)
    return loss_value, updated_non_trainable_variables


# 그래디언트 계산 함수
compute_gradients = jax.value_and_grad(compute_loss, has_aux=True)


# 트레이닝 스텝, Keras는 순수 함수형 optimizer.stateless_apply를 제공합니다
@jax.jit
def train_step(train_state, x, y):
    trainable_variables, non_trainable_variables, optimizer_variables = train_state
    (loss_value, non_trainable_variables), grads = compute_gradients(
        trainable_variables, non_trainable_variables, x, y
    )

    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )

    return loss_value, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )


# 모델과 옵티마이저 변수를 모든 장치에 복제
def get_replicated_train_state(devices):
    # 모든 변수는 모든 장치에서 복제됩니다.
    var_mesh = Mesh(devices, axis_names=("_"))
    # NamedSharding에서, 언급되지 않은 축은 복제됩니다. (여기서는 모든 축)
    var_replication = NamedSharding(var_mesh, P())

    # 모델 변수에 분산 설정 적용
    trainable_variables = jax.device_put(model.trainable_variables, var_replication)
    non_trainable_variables = jax.device_put(
        model.non_trainable_variables, var_replication
    )
    optimizer_variables = jax.device_put(optimizer.variables, var_replication)

    # 모든 상태를 하나의 튜플로 결합
    return (trainable_variables, non_trainable_variables, optimizer_variables)


num_devices = len(jax.local_devices())
print(f"Running on {num_devices} devices: {jax.local_devices()}")
devices = mesh_utils.create_device_mesh((num_devices,))

# 데이터는 배치 축을 따라 분할됩니다.
data_mesh = Mesh(devices, axis_names=("batch",))  # 메쉬 축의 이름 지정
data_sharding = NamedSharding(
    data_mesh,
    P(
        "batch",
    ),
)  # 샤딩된 파티션의 축 이름 지정

# 데이터 샤딩 표시
x, y = next(iter(train_data))
sharded_x = jax.device_put(x.numpy(), data_sharding)
print("Data sharding")
jax.debug.visualize_array_sharding(jax.numpy.reshape(sharded_x, [-1, 28 * 28]))

train_state = get_replicated_train_state(devices)

# 커스텀 트레이닝 루프
for epoch in range(num_epochs):
    data_iter = iter(train_data)
    for data in data_iter:
        x, y = data
        sharded_x = jax.device_put(x.numpy(), data_sharding)
        loss_value, train_state = train_step(train_state, sharded_x, y.numpy())
    print("Epoch", epoch, "loss:", loss_value)

# 모델 상태 업데이트 후 모델에 다시 기록
trainable_variables, non_trainable_variables, optimizer_variables = train_state
for variable, value in zip(model.trainable_variables, trainable_variables):
    variable.assign(value)
for variable, value in zip(model.non_trainable_variables, non_trainable_variables):
    variable.assign(value)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Running on 1 devices: [CpuDevice(id=0)]
Data sharding






                                     CPU 0





Epoch 0 loss: 0.28599858
Epoch 1 loss: 0.23666474
```

{{% /details %}}

이제 끝입니다!
