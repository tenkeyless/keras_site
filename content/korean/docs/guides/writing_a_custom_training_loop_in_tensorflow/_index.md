---
title: TensorFlow에서 처음부터 트레이닝 루프 작성하기
linkTitle: TensorFlow 트레이닝 루프
toc: true
weight: 9
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2019/03/01  
**{{< t f_last_modified >}}** 2023/06/25  
**{{< t f_description >}}** TensorFlow에서 낮은 레벨 트레이닝 및 평가 루프 작성하기.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/writing_a_custom_training_loop_in_tensorflow.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/writing_a_custom_training_loop_in_tensorflow.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 셋업 {#setup}

```python
import time
import os

# 이 가이드는 TensorFlow 백엔드에서만 실행할 수 있습니다.
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
import numpy as np
```

## 소개 {#introduction}

Keras는 기본 트레이닝 및 평가 루프인 `fit()`과 `evaluate()`를 제공합니다.
이러한 사용 방법은 {{< titledRelref "/docs/guides/training_with_built_in_methods/" >}} 가이드에서 다룹니다.

모델의 학습 알고리즘을 커스터마이즈하면서도 `fit()`의 편리함을 활용하고 싶다면
(예를 들어, `fit()`을 사용해 GAN을 트레이닝하려는 경우),
`Model` 클래스를 서브클래싱하고,
`fit()` 동안 반복적으로 호출되는 자체 `train_step()` 메서드를 구현할 수 있습니다.

이제, 트레이닝 및 평가에 대해 매우 낮은 레벨의 제어를 원한다면,
처음부터 직접 트레이닝 및 평가 루프를 작성해야 합니다.
이 가이드는 그것에 관한 것입니다.

## 첫 번째 엔드투엔드 예제 {#a-first-end-to-end-example}

간단한 MNIST 모델을 살펴봅시다:

```python
def get_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = get_model()
```

커스텀 트레이닝 루프를 사용하여, 미니 배치 그래디언트로 모델을 트레이닝해 봅시다.

먼저, 옵티마이저, 손실 함수, 데이터셋이 필요합니다:

```python
# 옵티마이저 인스턴스화
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# 손실 함수 인스턴스화
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 트레이닝 데이터셋 준비
batch_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# 검증을 위해 10,000개의 샘플을 예약합니다.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 트레이닝 데이터셋 준비
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# 검증 데이터셋 준비
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)
```

`GradientTape` scope 내에서 모델을 호출하면,
손실 값에 대해 레이어의 트레이닝 가능한 가중치의 그래디언트를 가져올 수 있습니다.
옵티마이저 인스턴스를 사용하여,
이 그래디언트를 사용해 (`model.trainable_weights`로 가져온) 이러한 변수를
업데이트할 수 있습니다.

다음은 단계별 트레이닝 루프입니다:

- 에포크를 반복하는 `for` 루프를 엽니다.
- 각 에포크에 대해, 데이터셋을 배치 단위로 반복하는 `for` 루프를 엽니다.
- 각 배치에 대해, `GradientTape()` scope를 엽니다.
- 이 scope 내에서, 모델을 호출(순전파)하고 손실을 계산합니다.
- scope 외부에서, 손실에 대한 모델 가중치의 그래디언트를 가져옵니다.
- 마지막으로, 옵티마이저를 사용해 그래디언트를 기반으로 모델의 가중치를 업데이트합니다.

```python
epochs = 3
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")

    # 데이터셋의 배치에 걸쳐 반복합니다.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # 순전파 동안 실행된 연산을 기록하기 위해, GradientTape를 엽니다.
        # 이를 통해 자동 미분이 가능합니다.
        with tf.GradientTape() as tape:
            # 레이어의 순전파를 실행합니다.
            # 레이어가 입력에 적용하는 연산은 GradientTape에 기록됩니다.
            logits = model(x_batch_train, training=True)  # 이 미니배치에 대한 로짓

            # 이 미니배치에 대한 손실 값을 계산합니다.
            loss_value = loss_fn(y_batch_train, logits)

        # 그레디언트 테이프를 사용해 손실에 대한
        # 트레이닝 가능한 변수의 그래디언트를 자동으로 가져옵니다.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # 그래디언트 하강법의 한 단계를 실행하여
        # 손실을 최소화하기 위해 변수의 값을 업데이트합니다.
        optimizer.apply(grads, model.trainable_weights)

        # 100 배치마다 로그를 출력합니다.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Start of epoch 0
Training loss (for 1 batch) at step 0: 95.3300
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 2.5622
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 3.1138
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.6748
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 1.3308
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 1.9813
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.8640
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 1.0696
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.3662
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.9556
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.7459
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.0468
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.7392
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.8435
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.3859
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.4156
Seen so far: 48032 samples
```

```plain
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.4045
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.5983
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.3154
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.7911
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.2607
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.2303
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.6048
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.7041
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.3669
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.6389
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.7739
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.3888
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.8133
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.2034
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.0768
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.1544
Seen so far: 48032 samples
```

```plain
Start of epoch 2
Training loss (for 1 batch) at step 0: 0.1250
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.0152
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.0917
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.1330
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.0884
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.2656
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.4375
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.2246
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.0748
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.1765
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.0130
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.4030
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.0667
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 1.0553
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.6513
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.0599
Seen so far: 48032 samples
```

{{% /details %}}

## 메트릭의 낮은 레벨 다루기 {#low-level-handling-of-metrics}

이 기본 루프에 메트릭 모니터링을 추가해 봅시다.

이렇게 처음부터 작성한 트레이닝 루프에서도,
빌트인 메트릭(또는 당신이 작성한 커스텀 메트릭)을 쉽게 재사용할 수 있습니다.
흐름은 다음과 같습니다:

- 루프 시작 시 메트릭을 인스턴스화합니다.
- 각 배치 후에 `metric.update_state()`를 호출합니다.
- 메트릭의 현재 값을 표시해야 할 때, `metric.result()`를 호출합니다.
- 메트릭의 상태를 초기화해야 할 때(일반적으로 에포크가 끝날 때),
  `metric.reset_state()`를 호출합니다.

이 지식을 사용하여,
각 에포크가 끝날 때 트레이닝 및 검증 데이터에 대한 `SparseCategoricalAccuracy`를 계산해 보겠습니다:

```python
# 새로운 모델 가져오기
model = get_model()

# 모델을 트레이닝할 옵티마이저 인스턴스화
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# 손실 함수 인스턴스화
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 메트릭 준비
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
```

트레이닝 및 평가 루프는 다음과 같습니다.

```python
epochs = 2
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    start_time = time.time()

    # 데이터셋의 배치에 걸쳐 반복합니다.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply(grads, model.trainable_weights)

        # 트레이닝 메트릭 업데이트.
        train_acc_metric.update_state(y_batch_train, logits)

        # 100 배치마다 로그를 출력합니다.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")

    # 각 에포크가 끝날 때 메트릭을 표시합니다.
    train_acc = train_acc_metric.result()
    print(f"Training acc over epoch: {float(train_acc):.4f}")

    # 각 에포크가 끝날 때 트레이닝 메트릭을 초기화합니다.
    train_acc_metric.reset_state()

    # 각 에포크가 끝날 때 검증 루프를 실행합니다.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # 검증 메트릭 업데이트
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f"Validation acc: {float(val_acc):.4f}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Start of epoch 0
Training loss (for 1 batch) at step 0: 89.1303
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 1.0351
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 2.9143
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 1.7842
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.9583
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 1.1100
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 2.1144
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.6801
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.6202
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 1.2570
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.3638
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 1.8402
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.7836
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.5147
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.4798
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.1653
Seen so far: 48032 samples
Training acc over epoch: 0.7961
Validation acc: 0.8825
Time taken: 46.06s
```

```plain
Start of epoch 1
Training loss (for 1 batch) at step 0: 1.3917
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.2600
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.7206
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.4987
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.3410
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.6788
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 1.1355
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1762
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.1801
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.3515
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.4344
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.2027
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.4649
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.6848
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.4594
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.3548
Seen so far: 48032 samples
Training acc over epoch: 0.8896
Validation acc: 0.9094
Time taken: 43.49s
```

{{% /details %}}

## [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)으로 트레이닝 스텝 속도 향상시키기 {#tffunction}

TensorFlow의 기본 런타임은 즉시 실행 모드(eager execution)입니다.
따라서, 위의 트레이닝 루프는 즉시(eagerly) 실행됩니다.

이는 디버깅에 유용하지만, 그래프 컴파일은 명확한 성능상의 이점을 가지고 있습니다.
계산을 정적 그래프로 설명하면, 프레임워크가 전역 성능 최적화를 적용할 수 있습니다.
프레임워크가 무엇이 다음에 올지 모르는 채로,
하나의 연산을 욕심껏 실행해야 하는 경우에는 불가능합니다.

텐서를 입력으로 받는 모든 함수는 정적 그래프로 컴파일할 수 있습니다.
다음과 같이 `@tf.function` 데코레이터를 추가하기만 하면 됩니다:

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)
    train_acc_metric.update_state(y, logits)
    return loss_value
```

평가 스텝에서도 동일한 작업을 해봅시다:

```python
@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
```

이제, 컴파일된 트레이닝 스텝을 사용하여 트레이닝 루프를 다시 실행해 봅시다:

```python
epochs = 2
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    start_time = time.time()

    # 데이터셋의 배치에 걸쳐 반복합니다.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        # 100 배치마다 로그를 출력합니다.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {float(loss_value):.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")

    # 각 에포크가 끝날 때 메트릭을 표시합니다.
    train_acc = train_acc_metric.result()
    print(f"Training acc over epoch: {float(train_acc):.4f}")

    # 각 에포크가 끝날 때 트레이닝 메트릭을 초기화합니다.
    train_acc_metric.reset_state()

    # 각 에포크가 끝날 때 검증 루프를 실행합니다.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f"Validation acc: {float(val_acc):.4f}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Start of epoch 0
Training loss (for 1 batch) at step 0: 0.5366
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.2732
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.2478
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.0263
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.4845
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.2239
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.2242
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.2122
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.2856
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.1957
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.2946
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.3080
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.2326
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.6514
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.2018
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.2812
Seen so far: 48032 samples
Training acc over epoch: 0.9104
Validation acc: 0.9199
Time taken: 5.73s
```

```plain
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.3080
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.3943
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.1657
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.1463
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.5359
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.1894
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.1801
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1724
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.3997
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.6017
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.1539
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.1078
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.8731
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.3110
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.6092
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.2046
Seen so far: 48032 samples
Training acc over epoch: 0.9189
Validation acc: 0.9358
Time taken: 3.17s
```

{{% /details %}}

Much faster, isn't it?

## 모델이 추적하는 손실의 낮은 레벨 다루기 {#low-level-handling-of-losses-tracked-by-the-model}

레이어와 모델은 순전파 중 `self.add_loss(value)`를 호출하는 레이어에 의해 생성된 모든 손실을 재귀적으로 추적합니다.
그 결과로 생성된 스칼라 손실 값들의 목록은 순전파가 끝난 후,
`model.losses` 속성을 통해 확인할 수 있습니다.

이러한 손실 요소들을 사용하고 싶다면,
이를 합산하여 트레이닝 스텝의 메인 손실에 추가해야 합니다.

다음은 활동 정규화 손실을 생성하는 레이어입니다:

```python
class ActivityRegularizationLayer(keras.layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs
```

간단한 모델을 만들어 사용해 봅시다:

```python
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu")(inputs)
# activity 정규화 레이어 삽입
x = ActivityRegularizationLayer()(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

현재의 트레이닝 스텝은 다음과 같이 생겼습니다:

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        # 순전파 동안 생성된 추가 손실을 더합니다.
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)
    train_acc_metric.update_state(y, logits)
    return loss_value
```

## 요약 {#summary}

이제 빌트인 트레이닝 루프를 사용하는 방법과,
직접 작성하는 방법에 대해 알아야 할 모든 것을 알게 되었습니다.

마지막으로, 이 가이드에서 배운 모든 내용을 결합한 간단한 엔드투엔드 예제를 소개합니다:
MNIST 숫자에 대해 트레이닝된 DCGAN입니다.

## 엔드투엔드 예제: 처음부터 작성하는 GAN 트레이닝 루프 {#end-to-end-example-a-gan-training-loop-from-scratch}

생성적 적대 신경망(Generative Adversarial Networks, GANs)에 대해 들어본 적이 있을 것입니다.
GANs는 이미지의 트레이닝 데이터셋(이미지의 "잠재 공간")의 잠재 분포를 학습하여,
거의 실제처럼 보이는 새로운 이미지를 생성할 수 있습니다.

GAN은 두 부분으로 구성됩니다:
잠재 공간의 점을 이미지 공간의 점으로 매핑하는 "생성자(generator)" 모델과,
실제 이미지(트레이닝 데이터셋에서 가져옴)와 가짜 이미지(생성기 네트워크의 출력)를 구분할 수 있는 분류기인 "판별자(discriminator)" 모델입니다.

GAN의 트레이닝 루프는 다음과 같은 형태입니다:

1. 판별자 트레이닝:
   - 잠재 공간에서 랜덤 포인트의 배치를 샘플링합니다.
   - "생성자" 모델을 통해, 포인트을 가짜 이미지로 변환합니다.
   - 실제 이미지의 배치를 얻고, 생성된 이미지와 결합합니다.
   - "판별자" 모델을 트레이닝하여, 생성된 이미지와 실제 이미지를 분류합니다.
2. 생성자 트레이닝:
   - 잠재 공간에서 랜덤 포인트를 샘플링합니다.
   - "생성자" 네트워크를 통해 포인트를 가짜 이미지로 변환합니다.
   - 실제 이미지의 배치를 얻고, 생성된 이미지와 결합합니다.
   - "생성자" 모델을 트레이닝하여, 판별자를 "속이고" 가짜 이미지를 실제 이미지로 분류하게 만듭니다.

GAN의 작동 방식에 대한 훨씬 더 자세한 개요는 [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)을 참조하세요.

이제 이 트레이닝 루프를 구현해 봅시다. 먼저, 가짜 숫자와 실제 숫자를 분류할 판별자를 생성합니다:

```python
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dense(1),
    ],
    name="discriminator",
)
discriminator.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "discriminator"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 14, 14, 64)        │        640 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu (LeakyReLU)         │ (None, 14, 14, 64)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (Conv2D)               │ (None, 7, 7, 128)         │     73,856 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_1 (LeakyReLU)       │ (None, 7, 7, 128)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ global_max_pooling2d            │ (None, 128)               │          0 │
│ (GlobalMaxPooling2D)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_6 (Dense)                 │ (None, 1)                 │        129 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 74,625 (291.50 KB)
 Trainable params: 74,625 (291.50 KB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

다음으로, 잠재 벡터를 `(28, 28, 1)` 모양의 출력(즉, MNIST 숫자)을 생성하는,
생성자 네트워크를 만들어 봅시다:

```python
latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        # 7x7x128 맵으로 reshape 하기 위해 128개의 계수를 생성합니다.
        keras.layers.Dense(7 * 7 * 128),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Reshape((7, 7, 128)),
        keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)
```

다음은 트레이닝 루프의 핵심 부분입니다.
보시다시피, 매우 간단합니다. 트레이닝 스텝 함수는 단 17줄로 구성되어 있습니다.

```python
# 생성자와 생성자를 위한 옵티마이저 각각을 인스턴스화합니다.
d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0004)

# 손실 함수를 인스턴스화합니다.
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step(real_images):
    # 잠재 공간에서 랜덤한 포인트를 샘플링합니다.
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    # 샘플을 가짜 이미지로 디코딩합니다.
    generated_images = generator(random_latent_vectors)
    # 가짜 이미지와 실제 이미지를 결합합니다.
    combined_images = tf.concat([generated_images, real_images], axis=0)

    # 가짜 이미지와 실제 이미지를 구분하는 레이블을 조립합니다.
    labels = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((real_images.shape[0], 1))], axis=0
    )
    # 레이블에 랜덤 노이즈를 추가합니다 - 중요한 트릭입니다!
    labels += 0.05 * tf.random.uniform(labels.shape)

    # 판별자를 트레이닝합니다.
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply(grads, discriminator.trainable_weights)

    # 잠재 공간에서 랜덤한 포인트를 샘플링합니다.
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    # "모든 이미지는 진짜 (all real images)"라고 말하는 레이블을 조립합니다.
    misleading_labels = tf.zeros((batch_size, 1))

    # 생성자를 트레이닝합니다.
    # (여기서 판별자의 가중치는 *절대로* 업데이트하면 안 됩니다)!
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(random_latent_vectors))
        g_loss = loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply(grads, generator.trainable_weights)
    return d_loss, g_loss, generated_images
```

이제 이미지 배치에 대해 `train_step`을 반복적으로 호출하여 우리의 GAN을 트레이닝해봅시다.

판별자와 생성자가 컨볼루션 신경망이기 때문에, 이 코드를 GPU에서 실행하는 것이 좋습니다.

```python
# 데이터셋 준비. 트레이닝 및 테스트 MNIST 숫자를 모두 사용합니다.
batch_size = 64
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

epochs = 1  # 실제로는 멋진 숫자를 생성하려면 최소 20 에포크가 필요합니다.
save_dir = "./"

for epoch in range(epochs):
    print(f"\nStart epoch {epoch}")

    for step, real_images in enumerate(dataset):
        # 한 배치의 실제 이미지에 대해 판별기 및 생성기를 트레이닝합니다.
        d_loss, g_loss, generated_images = train_step(real_images)

        # 로깅.
        if step % 100 == 0:
            # 메트릭 출력
            print(f"discriminator loss at step {step}: {d_loss:.2f}")
            print(f"adversarial loss at step {step}: {g_loss:.2f}")

            # 생성된 이미지 중 하나를 저장합니다.
            img = keras.utils.array_to_img(generated_images[0] * 255.0, scale=False)
            img.save(os.path.join(save_dir, f"generated_img_{step}.png"))

        # 실행 시간을 제한하기 위해 10 스텝 후에 중지합니다.
        # 실제로 모델을 트레이닝하려면, 아래의 줄을 제거하세요!
        if step > 10:
            break
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Start epoch 0
discriminator loss at step 0: 0.69
adversarial loss at step 0: 0.69
```

{{% /details %}}

이제 끝났습니다! Colab GPU에서 약 30초 동안 트레이닝한 후, 멋진 가짜 MNIST 숫자를 얻을 수 있습니다.
