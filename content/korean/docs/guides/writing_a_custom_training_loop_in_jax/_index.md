---
title: JAX에서 처음부터 트레이닝 루프 작성하기
linkTitle: JAX 트레이닝 루프
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2023/06/25  
**{{< t f_last_modified >}}** 2023/06/25  
**{{< t f_description >}}** JAX에서 낮은 레벨 트레이닝 및 평가 루프 작성하기.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/writing_a_custom_training_loop_in_jax.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/writing_a_custom_training_loop_in_jax.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 셋업 {#setup}

```python
import os

# 이 가이드는 jax 백엔드에서만 실행할 수 있습니다.
os.environ["KERAS_BACKEND"] = "jax"

import jax

# tf.data를 사용하기 위해 TF를 임포트합니다.
import tensorflow as tf
import keras
import numpy as np
```

## 소개 {#introduction}

Keras는 기본 트레이닝 및 평가 루프인 `fit()`과 `evaluate()`를 제공합니다.
이것들의 사용 방법은 {{< titledRelref "/docs/guides/training_with_built_in_methods" >}} 가이드에서 다룹니다.

모델의 학습 알고리즘을 커스터마이즈하면서도 `fit()`의 편리함을 활용하고 싶다면
(예를 들어, `fit()`을 사용해 GAN을 트레이닝하려는 경우),
`Model` 클래스를 서브클래싱하고,
`fit()` 동안 반복적으로 호출되는 자체 `train_step()` 메서드를 구현할 수 있습니다.

이제, 트레이닝 및 평가에 대해 매우 낮은 레벨의 제어를 원한다면,
처음부터 직접 트레이닝 및 평가 루프를 작성해야 합니다. 이 가이드는 그것에 관한 것입니다.

## 첫 번째 엔드투엔드 예제 {#a-first-end-to-end-example}

커스텀 트레이닝 루프를 작성하려면, 다음이 필요합니다:

- 트레이닝할 모델.
- 옵티마이저. `keras.optimizers`의 옵티마이저를 사용하거나, `optax` 패키지에서 사용할 수 있습니다.
- 손실 함수.
- 데이터셋. JAX 생태계의 표준은 [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data)를 통해 데이터를 로드하는 것이므로, 이를 사용할 것입니다.

이제 하나씩 설정해 보겠습니다.

먼저, 모델과 MNIST 데이터셋을 가져옵니다:

```python
def get_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = get_model()

# 트레이닝 데이터셋 준비.
batch_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)).astype("float32")
x_test = np.reshape(x_test, (-1, 784)).astype("float32")
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# 검증을 위해 10,000개의 샘플을 예약합니다.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 트레이닝 데이터셋 준비.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# 검증 데이터셋 준비.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)
```

다음으로, 손실 함수와 옵티마이저를 설정합니다. 이번에는 Keras 옵티마이저를 사용합니다.

```python
# 손실 함수 인스턴스화.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# 옵티마이저 인스턴스화.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
```

### JAX에서 그래디언트 얻기 {#getting-gradients-in-jax}

커스텀 트레이닝 루프를 사용하여, 미니 배치 그래디언트로 모델을 트레이닝해봅시다.

JAX에서는 *메타프로그래밍(metaprogramming)*을 통해 그래디언트를 계산합니다.
`jax.grad` (또는 `jax.value_and_grad`)를 함수에 호출하여,
그 함수에 대한 그래디언트 계산 함수를 생성합니다.

따라서 먼저 필요한 것은 손실 값을 반환하는 함수입니다.
이 함수를 사용하여 그래디언트 함수를 생성할 것입니다.
다음과 같은 형태입니다:

```python
def compute_loss(x, y):
    ...
    return loss
```

이와 같은 함수를 갖게 되면, 메타프로그래밍을 통해 다음과 같이 그래디언트를 계산할 수 있습니다:

```python
grad_fn = jax.grad(compute_loss)
grads = grad_fn(x, y)
```

일반적으로는, 단순히 그래디언트 값만 얻는 것이 아니라 손실 값도 함께 얻고자 합니다.
이를 위해 `jax.grad` 대신 `jax.value_and_grad`를 사용할 수 있습니다:

```python
grad_fn = jax.value_and_grad(compute_loss)
loss, grads = grad_fn(x, y)
```

### JAX 계산은 순수하게 stateless입니다. {#jax-computation-is-purely-stateless}

JAX에서는, 모든 것이 stateless 함수여야 하므로, 손실 계산 함수도 stateless여야 합니다.
이는 모든 Keras 변수(예: 가중치 텐서)를 함수의 입력으로 전달해야 하며,
순전파 동안 업데이트된 모든 변수를 함수의 출력으로 반환해야 함을 의미합니다.
함수는 부수 효과가 없어야 합니다.

순전파 동안, Keras 모델의 비트레이닝 변수는 업데이트될 수 있습니다.
이러한 변수는 예를 들어 RNG 시드 상태 변수나 BatchNormalization 통계일 수 있습니다.
우리는 이러한 변수들을 반환해야 합니다.
따라서 다음과 같은 함수가 필요합니다:

```python
def compute_loss_and_updates(trainable_variables, non_trainable_variables, x, y):
    ...
    return loss, non_trainable_variables
```

이러한 함수를 갖게 되면,
`value_and_grad`에서 `has_aux`를 지정하여 그래디언트 함수를 얻을 수 있습니다.
이는 JAX에 손실 계산 함수가 손실 외에도 더 많은 출력을 반환한다고 알려줍니다.
손실은 항상 첫 번째 출력이어야 한다는 점에 유의하세요.

```python
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
(loss, non_trainable_variables), grads = grad_fn(
    trainable_variables, non_trainable_variables, x, y
)
```

이제 기본 사항을 정립했으니, `compute_loss_and_updates` 함수를 구현해봅시다.
Keras 모델에는 `stateless_call` 메서드가 있는데,
이 메서드는 여기서 유용하게 사용할 수 있습니다.
`model.__call__`과 비슷하게 작동하지만,
모델의 모든 변수 값을 명시적으로 전달해야 하며,
`__call__` 출력뿐만 아니라 (잠재적으로 업데이트된) 트레이닝 불가능한(non-trainable) 변수도 반환합니다.

```python
def compute_loss_and_updates(trainable_variables, non_trainable_variables, x, y):
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x, training=True
    )
    loss = loss_fn(y, y_pred)
    return loss, non_trainable_variables
```

그래디언트 함수를 구해봅시다:

```python
grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
```

### 트레이닝 스텝 함수 {#the-training-step-function}

다음으로, 엔드투엔드 트레이닝 스텝을 구현해봅시다.
이 함수는 순전파를 실행하고, 손실을 계산하고, 그래디언트를 계산하며,
옵티마이저를 사용하여 트레이닝 가능한 변수를 업데이트하는 역할을 합니다.
이 함수도 stateless여야 하므로,
우리가 사용할 모든 상태 요소를 포함하는 `state` 튜플을 입력으로 받아야 합니다:

- `trainable_variables` 및 `non_trainable_variables`: 모델의 변수들.
- `optimizer_variables`: 옵티마이저의 상태 변수들, 예를 들어 모멘텀 누적기(momentum accumulators)와 같은 것들.

트레이닝 가능한 변수를 업데이트하기 위해,
옵티마이저의 stateless 메서드인 `stateless_apply`를 사용합니다.
이는 `optimizer.apply()`와 동등하지만,
항상 `trainable_variables`와 `optimizer_variables`를 전달해야 합니다.
이는 업데이트된 `trainable_variables`와 업데이트된 `optimizer_variables`를 반환합니다.

```python
def train_step(state, data):
    trainable_variables, non_trainable_variables, optimizer_variables = state
    x, y = data
    (loss, non_trainable_variables), grads = grad_fn(
        trainable_variables, non_trainable_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    # 업데이트된 상태 반환
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )
```

### `jax.jit` 빠르게 하기 {#make-it-fast-with-jaxjit}

기본적으로 JAX 연산은 TensorFlow eager 모드와 PyTorch eager 모드처럼, 즉시(eagerly) 실행됩니다.
그리고, TensorFlow eager 모드와 PyTorch eager 모드처럼 꽤 느립니다.
즉시 실행(eager) 모드는 디버깅 환경으로 더 적합하며,
실제 작업을 수행하는 방법으로는 적합하지 않습니다.
따라서, `train_step`을 컴파일하여 빠르게 만들어봅시다.

stateless JAX 함수가 있는 경우,
`@jax.jit` 데코레이터를 통해 이를 XLA로 컴파일할 수 있습니다.
첫 번째 실행 시 함수가 추적되고,
이후 실행에서는 추적된 그래프를 실행하게 됩니다.
(이는 `@tf.function(jit_compile=True)`와 유사합니다)
시도해 봅시다:

```python
@jax.jit
def train_step(state, data):
    trainable_variables, non_trainable_variables, optimizer_variables = state
    x, y = data
    (loss, non_trainable_variables), grads = grad_fn(
        trainable_variables, non_trainable_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    # 업데이트된 상태 반환
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )
```

이제 모델을 트레이닝할 준비가 되었습니다.
트레이닝 루프 자체는 간단합니다:
`loss, state = train_step(state, data)`를 반복적으로 호출하기만 하면 됩니다.

참고:

- [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)에서
  생성된(yielded) TF 텐서를 JAX 함수에 전달하기 전에 NumPy로 변환합니다.
- 모든 변수는 사전에 빌드되어야 합니다: 모델은 빌드되어야 하고, 옵티마이저도 빌드되어야 합니다.
  우리가 사용하는 것은 Functional API 모델이므로 이미 빌드되어 있지만,
  만약 서브클래스화된 모델이라면 데이터를 하나의 배치에 대해 호출하여 빌드해야 합니다.

```python
# 옵티마이저 변수 빌드
optimizer.build(model.trainable_variables)

trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
optimizer_variables = optimizer.variables
state = trainable_variables, non_trainable_variables, optimizer_variables

# 트레이닝 루프
for step, data in enumerate(train_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = train_step(state, data)
    # 100 배치마다 로그 출력
    if step % 100 == 0:
        print(f"Training loss (for 1 batch) at step {step}: {float(loss):.4f}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Training loss (for 1 batch) at step 0: 96.2726
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 2.0853
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.6535
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 1.2679
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.7563
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.7154
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 1.0267
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.6860
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.7306
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.4571
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.6023
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.9140
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.4224
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.6696
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.1399
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.5761
Seen so far: 48032 samples
```

{{% /details %}}

여기서 주목해야 할 중요한 점은 루프가 완전히 stateless하다는 것입니다.
모델에 첨부된 변수들(`model.weights`)은 루프 동안 절대 업데이트되지 않습니다.
변수의 새로운 값은 오직 `state` 튜플에만 저장됩니다.
이는 모델을 저장하기 전에, 새로운 변수 값을 다시 모델에 연결해야 한다는 것을 의미합니다.

업데이트하려는 각 모델 변수에 대해,
`variable.assign(new_value)`를 호출하기만 하면 됩니다:

```python
trainable_variables, non_trainable_variables, optimizer_variables = state
for variable, value in zip(model.trainable_variables, trainable_variables):
    variable.assign(value)
for variable, value in zip(model.non_trainable_variables, non_trainable_variables):
    variable.assign(value)
```

## 메트릭의 낮은 레벨 다루기 {#low-level-handling-of-metrics}

이 기본 트레이닝 루프에 메트릭 모니터링을 추가해 봅시다.

이렇게 처음부터 작성한 트레이닝 루프에서도
빌트인 Keras 메트릭(또는 사용자가 작성한 커스텀 메트릭)을 쉽게 재사용할 수 있습니다.
흐름은 다음과 같습니다:

- 루프 시작 시 메트릭을 인스턴스화합니다.
- `train_step` 인자와 `compute_loss_and_updates` 인자에 `metric_variables`를 포함시킵니다.
- `compute_loss_and_updates` 함수에서 `metric.stateless_update_state()`를 호출합니다.
  이는 `update_state()`와 같은 역할을 하지만, stateless인 것만 다릅니다.
- `train_step` 외부 (즉시 실행 범위(eager scope))에서 메트릭의 현재 값을 표시해야 할 때,
  새로운 메트릭 변수 값을 메트릭 객체에 연결하고, `metric.result()`를 호출합니다.
- 메트릭의 상태를 초기화해야 할 때(일반적으로 에포크가 끝날 때),
  `metric.reset_state()`를 호출합니다.

이 지식을 사용하여 트레이닝이 끝날 때,
트레이닝 및 검증 데이터에 대한 `CategoricalAccuracy`를 계산해 보겠습니다:

```python
# 새로운 모델 가져오기
model = get_model()

# 모델을 트레이닝할 옵티마이저 인스턴스화
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# 손실 함수 인스턴스화
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# 메트릭 준비
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()


def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, metric_variables, x, y
):
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    metric_variables = train_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, (non_trainable_variables, metric_variables)


grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)


@jax.jit
def train_step(state, data):
    (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    ) = state
    x, y = data
    (loss, (non_trainable_variables, metric_variables)), grads = grad_fn(
        trainable_variables, non_trainable_variables, metric_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    # 업데이트된 상태 반환
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    )
```

평가 스텝 함수도 준비해 보겠습니다:

```python
@jax.jit
def eval_step(state, data):
    trainable_variables, non_trainable_variables, metric_variables = state
    x, y = data
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    metric_variables = val_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, (
        trainable_variables,
        non_trainable_variables,
        metric_variables,
    )
```

다음은 우리의 루프입니다:

```python
# 옵티마이저 변수 빌드
optimizer.build(model.trainable_variables)

trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables
optimizer_variables = optimizer.variables
metric_variables = train_acc_metric.variables
state = (
    trainable_variables,
    non_trainable_variables,
    optimizer_variables,
    metric_variables,
)

# 트레이닝 루프
for step, data in enumerate(train_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = train_step(state, data)
    # 100 배치마다 로그 출력
    if step % 100 == 0:
        print(f"Training loss (for 1 batch) at step {step}: {float(loss):.4f}")
        _, _, _, metric_variables = state
        for variable, value in zip(train_acc_metric.variables, metric_variables):
            variable.assign(value)
        print(f"Training accuracy: {train_acc_metric.result()}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")

metric_variables = val_acc_metric.variables
(
    trainable_variables,
    non_trainable_variables,
    optimizer_variables,
    metric_variables,
) = state
state = trainable_variables, non_trainable_variables, metric_variables

# 평가 루프
for step, data in enumerate(val_dataset):
    data = (data[0].numpy(), data[1].numpy())
    loss, state = eval_step(state, data)
    # 100 배치마다 로그 출력
    if step % 100 == 0:
        print(f"Validation loss (for 1 batch) at step {step}: {float(loss):.4f}")
        _, _, metric_variables = state
        for variable, value in zip(val_acc_metric.variables, metric_variables):
            variable.assign(value)
        print(f"Validation accuracy: {val_acc_metric.result()}")
        print(f"Seen so far: {(step + 1) * batch_size} samples")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Training loss (for 1 batch) at step 0: 70.8851
Training accuracy: 0.09375
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 2.1930
Training accuracy: 0.6596534848213196
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 3.0249
Training accuracy: 0.7352300882339478
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.6004
Training accuracy: 0.7588247656822205
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 1.4633
Training accuracy: 0.7736907601356506
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 1.3367
Training accuracy: 0.7826846241950989
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.8767
Training accuracy: 0.7930532693862915
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.3479
Training accuracy: 0.8004636168479919
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.3608
Training accuracy: 0.8066869378089905
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.7582
Training accuracy: 0.8117369413375854
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 1.3135
Training accuracy: 0.8142170310020447
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 1.0202
Training accuracy: 0.8186308145523071
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.6766
Training accuracy: 0.822023332118988
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.7606
Training accuracy: 0.8257110118865967
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.7657
Training accuracy: 0.8290283679962158
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.6563
Training accuracy: 0.831653892993927
Seen so far: 48032 samples
Validation loss (for 1 batch) at step 0: 0.1622
Validation accuracy: 0.8329269289970398
Seen so far: 32 samples
Validation loss (for 1 batch) at step 100: 0.7455
Validation accuracy: 0.8338780999183655
Seen so far: 3232 samples
Validation loss (for 1 batch) at step 200: 0.2738
Validation accuracy: 0.836174488067627
Seen so far: 6432 samples
Validation loss (for 1 batch) at step 300: 0.1255
Validation accuracy: 0.8390461206436157
Seen so far: 9632 samples
```

{{% /details %}}

## 모델이 추적하는 손실의 낮은 레벨 다루기 {#low-level-handling-of-losses-tracked-by-the-model}

레이어와 모델은 순전파 중,
`self.add_loss(value)`를 호출하는 레이어에 의해 생성된 모든 손실을 재귀적으로 추적합니다.
그 결과 생성된 스칼라 손실 값들의 목록은,
순전파가 끝난 후 `model.losses` 속성을 통해 확인할 수 있습니다.

이러한 손실 요소들을 사용하고 싶다면,
이를 합산하여 트레이닝 스텝의 메인 손실에 추가해야 합니다.

활동 정규화 손실을 생성하는 다음 레이어를 고려해 보세요:

```python
class ActivityRegularizationLayer(keras.layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * jax.numpy.sum(inputs))
        return inputs
```

간단한 모델을 빌드해봅시다:

```python
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu")(inputs)
# activity 정규화 레이어를 추가합니다.
x = ActivityRegularizationLayer()(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

다음은 `compute_loss_and_updates` 함수가 어떻게 생겨야 하는지 보여줍니다:

- `model.stateless_call()`에 `return_losses=True`를 전달합니다.
- 결과로 생성된 `losses`를 합산하여, 메인 손실에 추가합니다.

```python
def compute_loss_and_updates(
    trainable_variables, non_trainable_variables, metric_variables, x, y
):
    y_pred, non_trainable_variables, losses = model.stateless_call(
        trainable_variables, non_trainable_variables, x, return_losses=True
    )
    loss = loss_fn(y, y_pred)
    if losses:
        loss += jax.numpy.sum(losses)
    metric_variables = train_acc_metric.stateless_update_state(
        metric_variables, y, y_pred
    )
    return loss, non_trainable_variables, metric_variables
```

이것으로 끝입니다!
