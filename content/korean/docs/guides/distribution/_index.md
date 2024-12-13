---
title: Keras 3으로 분산 트레이닝하기
linkTitle: Keras 3 분산 트레이닝
toc: true
weight: 18
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [Qianli Zhu](https://github.com/qlzh727)  
**{{< t f_date_created >}}** 2023/11/07  
**{{< t f_last_modified >}}** 2023/11/07  
**{{< t f_description >}}** 멀티 백엔드 Keras를 위한 분산 API에 대한 완벽 가이드.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/distribution.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/distribution.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

Keras 분산 API는 JAX, TensorFlow, PyTorch와 같은
다양한 백엔드에서 분산 딥러닝을 용이하게 하기 위해 설계된 새로운 인터페이스입니다.
이 강력한 API는 데이터 및 모델 병렬 처리를 가능하게 하는 도구 모음을 제공하며,
여러 가속기 및 호스트에서 딥러닝 모델을 효율적으로 확장할 수 있습니다.
GPU나 TPU의 성능을 활용하든, API는
분산 환경 초기화, 디바이스 메시 정의, 텐서의 계산 리소스 간 레이아웃 조정을 위한 간소화된 접근 방식을 제공합니다.
`DataParallel` 및 `ModelParallel`과 같은 클래스를 통해,
병렬 계산의 복잡성을 추상화하여,
개발자가 기계 학습 워크플로우를 가속화하는 것을 보다 쉽게 만듭니다.

## 작동 방식 {#how-it-works}

Keras 분산 API는 글로벌 프로그래밍 모델을 제공하여,
개발자가 마치 단일 장치에서 작업하는 것처럼,
글로벌 컨텍스트에서 텐서 연산을 구성할 수 있게 합니다.
이 과정에서 API는 여러 장치에 걸쳐 분산을 자동으로 관리합니다.
API는 기본 프레임워크(JAX 등)를 활용하여,
프로그램과 텐서를 샤딩 지시어(sharding directives)에 따라 분산시키며,
이 과정을 단일 프로그램 다중 데이터(SPMD, single program, multiple data) 확장이라고 합니다.

응용 프로그램을 샤딩 지시어와 분리함으로써,
API는 동일한 응용 프로그램을 단일 장치, 다중 장치 또는 여러 클라이언트에서 실행할 수 있도록 하며,
글로벌 시맨틱을 유지합니다.

## 셋업 {#setup}

```python
import os

# 분산 API는 현재 JAX 백엔드에서만 구현되어 있습니다.
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers
import jax
import numpy as np
from tensorflow import data as tf_data  # 데이터셋 입력을 위한 모듈.
```

## `DeviceMesh` 및 `TensorLayout` {#devicemesh-and-tensorlayout}

Keras 분산 API의 [`keras.distribution.DeviceMesh`]({{< relref "/docs/api/distribution/layout_map#devicemesh-class" >}}) 클래스는 분산 계산을 위해 구성된, 컴퓨팅 장치 클러스터를 나타냅니다.
이는 [`jax.sharding.Mesh`](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh) 및 [`tf.dtensor.Mesh`](https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Mesh)와 유사한 개념으로,
물리적 장치를 논리적 메쉬 구조에 매핑하는 데 사용됩니다.

그 후 `TensorLayout` 클래스는 지정된 축을 따라
텐서가 어떻게 `DeviceMesh`에 분산되는지 명시하며,
이 축들은 `DeviceMesh`의 축 이름과 일치합니다.

더 자세한 개념 설명은, [TensorFlow DTensor 가이드](https://www.tensorflow.org/guide/dtensor_overview#dtensors_model_of_distributed_tensors)에서 확인할 수 있습니다.

```python
# 로컬에서 사용 가능한 GPU 장치를 검색합니다.
devices = jax.devices("gpu")  # 8개의 로컬 GPU가 있다고 가정합니다.

# 데이터 및 모델 병렬 축을 가진 2x4 장치 메쉬를 정의합니다.
mesh = keras.distribution.DeviceMesh(
    shape=(2, 4), axis_names=["data", "model"], devices=devices
)

# 텐서가 메쉬에 어떻게 분산되는지 설명하는 2D 레이아웃.
# 레이아웃은 "model"을 행으로, "data"를 열로 하는 2D 그리드로 시각화할 수 있으며,
# 물리적 장치에 매핑될 때 [4, 2] 그리드입니다.
layout_2d = keras.distribution.TensorLayout(axes=("model", "data"), device_mesh=mesh)

# 이미지 입력의 데이터 병렬에 사용할 수 있는 4D 레이아웃.
replicated_layout_4d = keras.distribution.TensorLayout(
    axes=("data", None, None, None), device_mesh=mesh
)
```

## `Distribution` {#distribution}

Keras의 `Distribution` 클래스는 커스텀 분산 전략을 개발하기 위한 기초적인 추상 클래스입니다.
이 클래스는 장치 메쉬(device mesh)에 걸쳐
모델의 변수, 입력 데이터 및 중간 계산을 분산하는 핵심 로직을 캡슐화합니다.
일반 사용자로서는 이 클래스를 직접 사용할 필요가 없으며,
대신 `DataParallel` 또는 `ModelParallel`과 같은 하위 클래스를 사용하게 됩니다.

## `DataParallel` {#dataparallel}

`DataParallel` 클래스는 Keras 분산 API에서
분산 트레이닝의 데이터 병렬 처리를 위한 전략으로 설계되었으며,
`DeviceMesh` 내 모든 장치에 걸쳐 모델 가중치가 복제되고,
각 장치가 입력 데이터의 일부를 처리하는 방식입니다.

다음은 `DataParallel` 클래스를 사용하는 예시입니다.

```python
# 장치 목록을 사용하여 DataParallel 생성.
# shortcut으로서 장치 목록을 생략하면,
# Keras가 모든 로컬 가용 장치를 자동으로 감지합니다.
# 예: data_parallel = DataParallel()
data_parallel = keras.distribution.DataParallel(devices=devices)

# 또는 1D `DeviceMesh`로 DataParallel을 생성할 수 있습니다.
mesh_1d = keras.distribution.DeviceMesh(
    shape=(8,), axis_names=["data"], devices=devices
)
data_parallel = keras.distribution.DataParallel(device_mesh=mesh_1d)

inputs = np.random.normal(size=(128, 28, 28, 1))
labels = np.random.normal(size=(128, 10))
dataset = tf_data.Dataset.from_tensor_slices((inputs, labels)).batch(16)

# 글로벌 분산 설정.
keras.distribution.set_distribution(data_parallel)

# 여기서부터 모든 모델 가중치는 `DeviceMesh`의 모든 장치에 복제됩니다.
# 여기에는 RNG 상태, 옵티마이저 상태, 메트릭 등이 포함됩니다.
# `model.fit` 또는 `model.evaluate`에 입력되는 데이터셋은,
# 배치 차원에서 고르게 분할되어 모든 장치로 전달됩니다.
# 수동으로 손실을 집계할 필요가 없습니다.
# 모든 계산은 전역 컨텍스트에서 수행됩니다.
inputs = layers.Input(shape=(28, 28, 1))
y = layers.Flatten()(inputs)
y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
y = layers.Dropout(0.4)(y)
y = layers.Dense(units=10, activation="softmax")(y)
model = keras.Model(inputs=inputs, outputs=y)

model.compile(loss="mse")
model.fit(dataset, epochs=3)
model.evaluate(dataset)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/3
 8/8 ━━━━━━━━━━━━━━━━━━━━ 8s 30ms/step - loss: 1.0116
Epoch 2/3
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.9237
Epoch 3/3
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.8736
 8/8 ━━━━━━━━━━━━━━━━━━━━ 5s 5ms/step - loss: 0.8349

0.842325747013092
```

{{% /details %}}

## `ModelParallel` 및 `LayoutMap` {#modelparallel-and-layoutmap}

`ModelParallel`은 모델 가중치가 단일 가속기에 맞지 않을 때 유용하게 사용됩니다.
이 설정을 통해 모델 가중치나 활성화 텐서를 `DeviceMesh`의 모든 장치에 분산하여,
대규모 모델을 수평적으로 확장할 수 있습니다.

`DataParallel` 모델에서는 모든 가중치가 완전히 복제되지만,
`ModelParallel`에서는 최적의 성능을 위해 가중치 레이아웃을 커스터마이즈해야 할 때가 많습니다.
이를 위해 `LayoutMap`을 사용하여,
글로벌 관점에서 모든 가중치와 중간 텐서에 대해, `TensorLayout`을 지정할 수 있습니다.

`LayoutMap`은 문자열을 `TensorLayout` 인스턴스와 매핑하는 dict와 유사한 객체입니다.
일반적인 Python dict와 달리, 문자열 키는 값을 검색할 때 정규식으로 처리됩니다.
이 클래스는 `TensorLayout`의 명명 규칙을 정의하고,
해당하는 `TensorLayout` 인스턴스를 검색할 수 있도록 합니다.
일반적으로, 검색에 사용되는 키는 변수의 식별자인 `variable.path` 속성입니다.
shortcut으로서, 튜플이나 축 이름 목록도 값 삽입 시 허용되며,
자동으로 `TensorLayout`으로 변환됩니다.

`LayoutMap`은 선택적으로 `DeviceMesh`를 포함하여,
`TensorLayout.device_mesh`를 설정하지 않았을 때, 자동으로 이를 채울 수 있습니다.
키가 정확히 일치하지 않으면, 기존 키들이 정규식으로 간주되어 입력 키와 다시 비교됩니다.
다수의 매치가 있을 경우 `ValueError`가 발생하고, 매치가 없으면 `None`을 반환합니다.

```python
mesh_2d = keras.distribution.DeviceMesh(
    shape=(2, 4), axis_names=["data", "model"], devices=devices
)
layout_map = keras.distribution.LayoutMap(mesh_2d)

# 아래 규칙은 d1/kernel에 매치되는 모든 가중치가 모델 차원(4개의 장치)으로
# 분할된다는 의미입니다. d1/bias도 동일하게 설정됩니다.
# 그 외의 모든 가중치는 완전히 복제됩니다.
layout_map["d1/kernel"] = (None, "model")
layout_map["d1/bias"] = ("model",)

# 레이어 출력의 레이아웃도 설정할 수 있습니다.
layout_map["d2/output"] = ("data", None)

model_parallel = keras.distribution.ModelParallel(layout_map, batch_dim_name="data")

keras.distribution.set_distribution(model_parallel)

inputs = layers.Input(shape=(28, 28, 1))
y = layers.Flatten()(inputs)
y = layers.Dense(units=200, use_bias=False, activation="relu", name="d1")(y)
y = layers.Dropout(0.4)(y)
y = layers.Dense(units=10, activation="softmax", name="d2")(y)
model = keras.Model(inputs=inputs, outputs=y)

# 데이터는 2개의 장치가 있는 "data" 차원으로 분할됩니다.
model.compile(loss="mse")
model.fit(dataset, epochs=3)
model.evaluate(dataset)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/3

/opt/conda/envs/keras-jax/lib/python3.10/site-packages/jax/_src/interpreters/mlir.py:761: UserWarning: Some donated buffers were not usable: ShapedArray(float32[784,50]).
See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer-donation.
  warnings.warn("Some donated buffers were not usable:"

 8/8 ━━━━━━━━━━━━━━━━━━━━ 5s 8ms/step - loss: 1.0266
Epoch 2/3
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.9181
Epoch 3/3
 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.8725
 8/8 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 0.8381

0.8502610325813293
```

{{% /details %}}

메시 구조를 변경하여 더 많은 데이터 병렬 처리나 모델 병렬 처리 간의 계산을 조정하는 것도 쉽습니다.
메시의 shape을 조정함으로써 이를 할 수 있습니다. 다른 코드에 대한 변경은 필요하지 않습니다.

```python
full_data_parallel_mesh = keras.distribution.DeviceMesh(
    shape=(8, 1), axis_names=["data", "model"], devices=devices
)
more_data_parallel_mesh = keras.distribution.DeviceMesh(
    shape=(4, 2), axis_names=["data", "model"], devices=devices
)
more_model_parallel_mesh = keras.distribution.DeviceMesh(
    shape=(2, 4), axis_names=["data", "model"], devices=devices
)
full_model_parallel_mesh = keras.distribution.DeviceMesh(
    shape=(1, 8), axis_names=["data", "model"], devices=devices
)
```

### 추가 자료 {#further-reading}

1.  [JAX 분산 배열 및 자동 병렬화](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
2.  [JAX sharding 모듈](https://jax.readthedocs.io/en/latest/jax.sharding.html)
3.  [TensorFlow DTensors를 사용한 분산 트레이닝](https://www.tensorflow.org/tutorials/distribute/dtensor_ml_tutorial)
4.  [TensorFlow DTensor 개념](https://www.tensorflow.org/guide/dtensor_overview)
5.  [tf.keras로 DTensors 사용하기](https://www.tensorflow.org/tutorials/distribute/dtensor_keras_tutorial)
