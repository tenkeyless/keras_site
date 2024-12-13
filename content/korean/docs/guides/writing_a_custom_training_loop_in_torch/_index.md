---
title: PyTorch에서 처음부터 트레이닝 루프 작성하기
linkTitle: PyTorch 트레이닝 루프
toc: true
weight: 10
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2023/06/25  
**{{< t f_last_modified >}}** 2023/06/25  
**{{< t f_description >}}** PyTorch에서 낮은 레벨 트레이닝 및 평가 루프 작성하기.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/writing_a_custom_training_loop_in_torch.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/writing_a_custom_training_loop_in_torch.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 셋업 {#setup}

```python
import os

# 이 가이드는 torch 백엔드에서만 실행할 수 있습니다.
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
import numpy as np
```

## 소개 {#introduction}

Keras는 `fit()`과 `evaluate()`라는 기본 트레이닝 및 평가 루프를 제공합니다.
이들의 사용법은 ({{< titledRelref "/docs/guides/training_with_built_in_methods" >}}) 가이드에서 다룹니다.

모델의 학습 알고리즘을 커스터마이즈하면서도 `fit()`의 편리함을 활용하고 싶다면
(예를 들어, `fit()`을 사용해 GAN을 트레이닝하려는 경우),
`Model` 클래스를 서브클래싱하고,
`fit()` 동안 반복적으로 호출되는 `train_step()` 메서드를 직접 구현할 수 있습니다.

이제, 트레이닝 및 평가를 아주 낮은 레벨에서 제어하려면,
처음부터 직접 트레이닝 및 평가 루프를 작성해야 합니다.
이 가이드는 바로 이러한 방법에 대해 설명합니다.

## 첫 번째 엔드투엔드 예제 {#a-first-end-to-end-example}

커스텀 트레이닝 루프를 작성하려면, 다음과 같은 요소들이 필요합니다:

- 물론, 트레이닝할 모델.
- 옵티마이저.
  - `keras.optimizers`의 옵티마이저나 `torch.optim`의 파이토치 네이티브 옵티마이저 중 하나를 사용할 수 있습니다.
- 손실 함수.
  - `keras.losses`의 손실 함수나 `torch.nn`의 파이토치 네이티브 손실 함수를 사용할 수 있습니다.
- 데이터셋.
  - [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), 파이토치 `DataLoader`, 파이썬 생성기 등 어떤 형식도 사용할 수 있습니다.

이 요소들을 정리해보겠습니다.
각 경우에 대해 파이토치 네이티브 객체를 사용할 것입니다 — 단, 물론 Keras 모델은 제외하고요.

먼저, 모델과 MNIST 데이터셋을 가져와 보겠습니다:

```python
# 간단한 MNIST 모델을 고려해 봅시다.
def get_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# MNIST 데이터셋을 로드하고 이를 파이토치 DataLoader에 넣습니다
# 트레이닝 데이터셋을 준비합니다.
batch_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)).astype("float32")
x_test = np.reshape(x_test, (-1, 784)).astype("float32")
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# 10,000개의 샘플을 검증용으로 예약합니다.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 파이토치 Datasets 생성
train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_train), torch.from_numpy(y_train)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_val), torch.from_numpy(y_val)
)

# Datasets을 위한 DataLoaders 생성
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
```

다음은 PyTorch 옵티마이저와 PyTorch 손실 함수입니다:

```python
# 파이토치 옵티마이저 인스턴스화
model = get_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 파이토치 손실 함수 인스턴스화
loss_fn = torch.nn.CrossEntropyLoss()
```

이제 커스텀 트레이닝 루프를 사용하여, 미니 배치 그래디언트로 모델을 트레이닝해봅시다.

손실 텐서에 대해 `loss.backward()`를 호출하면 역전파가 실행됩니다.
이것이 완료되면, 당신의 옵티마이저는 각 변수에 대한 그래디언트를 마술처럼 알 수 있고,
이를 통해 `optimizer.step()`를 사용하여 변수들을 업데이트할 수 있습니다.
텐서, 변수, 옵티마이저는 모두 숨겨진 전역 상태를 통해 서로 연결되어 있습니다.
또한 `loss.backward()`를 호출하기 전에,
`model.zero_grad()`를 호출하는 것을 잊지 마세요.
그렇지 않으면, 변수에 대한 올바른 그래디언트를 얻을 수 없습니다.

다음은 단계별로 설명한, 트레이닝 루프입니다:

- 에포크를 반복하는 `for` 루프를 엽니다.
- 각 에포크마다, 데이터셋을 배치 단위로 반복하는 `for` 루프를 엽니다.
- 각 배치마다, 입력 데이터를 모델에 전달하여 예측값을 얻은 다음, 이를 사용해 손실 값을 계산합니다.
- `loss.backward()`를 호출합니다.
- scope 바깥에서, 손실에 대한 모델 가중치의 그래디언트를 가져옵니다.
- 마지막으로, 옵티마이저를 사용하여 그래디언트를 기반으로 모델의 가중치를 업데이트합니다.

```python
epochs = 3
for epoch in range(epochs):
    for step, (inputs, targets) in enumerate(train_dataloader):
        # 순전파 (Forward pass)
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        # 역전파 (Backward pass)
        model.zero_grad()
        loss.backward()

        # 옵티마이저 변수 업데이트
        optimizer.step()

        # # 100개 배치마다 로그 출력.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Training loss (for 1 batch) at step 0: 110.9115
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 2.9493
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 2.7383
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 1.6616
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 1.5927
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 1.0992
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.5425
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.3308
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.8231
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.5570
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.6321
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.4962
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 1.0833
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 1.3607
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 1.1250
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 1.2562
Seen so far: 48032 samples
Training loss (for 1 batch) at step 0: 0.5181
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.3939
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.3406
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.1122
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.2015
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.1184
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 1.0702
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.4062
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.4570
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 1.2490
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.0714
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.3677
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.8291
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.8320
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.1179
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.5390
Seen so far: 48032 samples
Training loss (for 1 batch) at step 0: 0.1309
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.4061
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.2734
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.2972
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.4282
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3504
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.3556
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.7834
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.2522
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.2056
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.3259
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.5215
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.8051
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.4423
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.0473
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.1419
Seen so far: 48032 samples
```

{{% /details %}}

다른 방법으로, Keras 옵티마이저와 Keras 손실 함수를 사용할 때의 루프가 어떻게 보이는지 살펴보겠습니다.

중요한 차이점:

- 각 트레이닝 가능한 변수에 대해 호출되는, `v.value.grad`를 통해 변수의 그래디언트를 검색합니다.
- `optimizer.apply()`를 통해 변수들을 업데이트하는데, 이는 반드시 `torch.no_grad()` scope 내에서 호출되어야 합니다.

**또한 중요한 주의사항:** NumPy/TensorFlow/JAX/Keras API뿐만 아니라 Python의 `unittest` API는 모두,
`fn(y_true, y_pred)` (참고값이 먼저, 예측값이 두 번째) 순서의 인자 순서를 사용하지만,
PyTorch는 실제로 손실에 대해 `fn(y_pred, y_true)` 순서를 사용합니다.
따라서, `logits`와 `targets`의 순서를 바꿔야 합니다.

```python
model = get_model()
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    for step, (inputs, targets) in enumerate(train_dataloader):
        # 순전파 (Forward pass)
        logits = model(inputs)
        loss = loss_fn(targets, logits)

        # 역전파 (Backward pass)
        model.zero_grad()
        trainable_weights = [v for v in model.trainable_weights]

        # 손실에 대해 torch.Tensor.backward()를 호출하여
        # 가중치에 대한 그래디언트를 계산합니다.
        loss.backward()
        gradients = [v.value.grad for v in trainable_weights]

        # 가중치 업데이트
        with torch.no_grad():
            optimizer.apply(gradients, trainable_weights)

        # 100개 배치마다 로그 출력.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Start of epoch 0
Training loss (for 1 batch) at step 0: 98.9569
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 5.3304
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.3246
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 1.6745
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 1.0936
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 1.4159
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.2796
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 2.3532
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.7533
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 1.0432
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.3959
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.4722
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.3851
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.8599
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.1237
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.4919
Seen so far: 48032 samples
```

```plain
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.8972
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.5844
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.1285
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.0671
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.4296
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.1483
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.0230
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1368
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.1531
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.0472
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.2343
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.4449
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.3942
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.3236
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.0717
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.9288
Seen so far: 48032 samples
```

```plain
Start of epoch 2
Training loss (for 1 batch) at step 0: 0.9393
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.2383
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.1116
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.6736
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.6713
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3394
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.2385
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.4248
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.0200
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.1259
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.7566
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.0594
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.2821
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.2088
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.5654
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.0512
Seen so far: 48032 samples
```

{{% /details %}}

## 메트릭의 낮은 레벨 다루기 {#low-level-handling-of-metrics}

이 기본 트레이닝 루프에 메트릭 모니터링을 추가해봅시다.

이와 같이 처음부터 작성한 트레이닝 루프에서,
Keras의 빌트인 메트릭(또는 직접 작성한 커스텀 메트릭)을 쉽게 재사용할 수 있습니다.
흐름은 다음과 같습니다:

- 루프 시작 시 메트릭 인스턴스화
- 각 배치 후 `metric.update_state()` 호출
- 메트릭의 현재 값을 표시해야 할 때, `metric.result()` 호출
- 메트릭의 상태를 초기화해야 할 때(일반적으로 에포크가 끝날 때), `metric.reset_state()` 호출

이 지식을 사용하여 각 에포크가 끝날 때,
트레이닝 및 검증 데이터에 대해 `CategoricalAccuracy`를 계산해봅시다:

```python
# 새로운 모델 가져오기
model = get_model()

# 모델을 트레이닝할 옵티마이저 인스턴스화.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# 손실 함수 인스턴스화.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# 메트릭 준비.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()
```

다음은 우리의 트레이닝 및 평가 루프입니다:

```python
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    for step, (inputs, targets) in enumerate(train_dataloader):
        # 순전파 (Forward pass)
        logits = model(inputs)
        loss = loss_fn(targets, logits)

        # 역전파 (Backward pass)
        model.zero_grad()
        trainable_weights = [v for v in model.trainable_weights]

        # 손실에 대해 torch.Tensor.backward()를 호출하여
        # 가중치에 대한 그래디언트를 계산합니다.
        loss.backward()
        gradients = [v.value.grad for v in trainable_weights]

        # 가중치 업데이트
        with torch.no_grad():
            optimizer.apply(gradients, trainable_weights)

        # 트레이닝 메트릭 업데이트.
        train_acc_metric.update_state(targets, logits)

        # 100개 배치마다 로그 출력.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")

    # 각 에포크가 끝날 때 메트릭 표시.
    train_acc = train_acc_metric.result()
    print(f"Training acc over epoch: {float(train_acc):.4f}")

    # 각 에포크가 끝날 때 트레이닝 메트릭 초기화
    train_acc_metric.reset_state()

    # 각 에포크가 끝날 때 검증 루프 실행.
    for x_batch_val, y_batch_val in val_dataloader:
        val_logits = model(x_batch_val, training=False)
        # 검증 메트릭 업데이트
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f"Validation acc: {float(val_acc):.4f}")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Start of epoch 0
Training loss (for 1 batch) at step 0: 59.2206
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 8.9801
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 5.2990
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 3.6978
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 1.9965
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 2.1896
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 1.2416
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.9403
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.1838
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.5884
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.7836
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.7015
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.3335
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.2763
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.4787
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.2562
Seen so far: 48032 samples
Training acc over epoch: 0.8411
Validation acc: 0.8963
```

```plain
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.3417
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 1.1465
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.7274
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.1273
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.6500
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.2008
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.7483
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.5821
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.5696
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.3112
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.1761
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.1811
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.2736
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.3848
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.4627
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.3934
Seen so far: 48032 samples
Training acc over epoch: 0.9053
Validation acc: 0.9221
```

```plain
Start of epoch 2
Training loss (for 1 batch) at step 0: 0.5743
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.4448
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.9880
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.2268
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.5607
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.1178
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.4305
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1712
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.3109
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.1548
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.1090
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.5169
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.3791
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.6963
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.6204
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.1111
Seen so far: 48032 samples
Training acc over epoch: 0.9216
Validation acc: 0.9356
```

{{% /details %}}

## 모델에 의해 추적되는 손실의 낮은 레벨 다루기 {#low-level-handling-of-losses-tracked-by-the-model}

레이어와 모델은 순전파 중 `self.add_loss(value)`를 호출하는 레이어에 의해 생성된 모든 손실을 재귀적으로 추적합니다.
이렇게 생성된 스칼라 손실 값 목록은 순전파가 끝날 때 `model.losses` 속성을 통해 확인할 수 있습니다.

이러한 손실 구성 요소를 사용하려면, 이를 합산하여 트레이닝 스텝에서 메인 손실에 추가해야 합니다.

다음은 activity 정규화 손실을 생성하는 레이어를 고려해봅시다:

```python
class ActivityRegularizationLayer(keras.layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * torch.sum(inputs))
        return inputs
```

이 레이어를 사용하는 정말 간단한 모델을 만들어 봅시다:

```python
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu")(inputs)
# 레이어로서 activity 정규화 삽입
x = ActivityRegularizationLayer()(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

이제 우리의 트레이닝 루프는 다음과 같이 되어야 합니다:

```python
# 새로운 모델 가져오기
model = get_model()

# 모델을 트레이닝할 옵티마이저 인스턴스화.
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# 손실 함수 인스턴스화.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

# 메트릭 준비.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    for step, (inputs, targets) in enumerate(train_dataloader):
        # 순전파 (Forward pass)
        logits = model(inputs)
        loss = loss_fn(targets, logits)
        if model.losses:
            loss = loss + torch.sum(*model.losses)

        # 역전파 (Backward pass)
        model.zero_grad()
        trainable_weights = [v for v in model.trainable_weights]

        # 손실에 대해 torch.Tensor.backward()를 호출하여
        # 가중치에 대한 그래디언트를 계산합니다.
        loss.backward()
        gradients = [v.value.grad for v in trainable_weights]

        # 가중치 업데이트
        with torch.no_grad():
            optimizer.apply(gradients, trainable_weights)

        # 트레이닝 메트릭 업데이트.
        train_acc_metric.update_state(targets, logits)

        # 100개 배치마다 로그 출력.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {loss.detach().numpy():.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")

    # 각 에포크가 끝날 때 메트릭 표시.
    train_acc = train_acc_metric.result()
    print(f"Training acc over epoch: {float(train_acc):.4f}")

    # 각 에포크가 끝날 때 트레이닝 메트릭 초기화
    train_acc_metric.reset_state()

    # 각 에포크가 끝날 때 검증 루프 실행.
    for x_batch_val, y_batch_val in val_dataloader:
        val_logits = model(x_batch_val, training=False)
        # 검증 메트릭 업데이트
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_state()
    print(f"Validation acc: {float(val_acc):.4f}")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Start of epoch 0
Training loss (for 1 batch) at step 0: 138.7979
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 4.4268
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 1.0779
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 1.7229
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.5801
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.4298
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.4717
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 1.3369
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 1.3239
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.5972
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.1983
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.5228
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 1.0025
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.3424
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.5196
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.4287
Seen so far: 48032 samples
Training acc over epoch: 0.8089
Validation acc: 0.8947
```

```plain
Start of epoch 1
Training loss (for 1 batch) at step 0: 0.2903
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.4118
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 0.6533
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.0402
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.3638
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3313
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.5119
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1628
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.4793
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.2726
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.5721
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.5783
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.2533
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.2218
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.1232
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.6805
Seen so far: 48032 samples
Training acc over epoch: 0.8970
Validation acc: 0.9097
```

```plain
Start of epoch 2
Training loss (for 1 batch) at step 0: 0.4553
Seen so far: 32 samples
Training loss (for 1 batch) at step 100: 0.3975
Seen so far: 3232 samples
Training loss (for 1 batch) at step 200: 1.2382
Seen so far: 6432 samples
Training loss (for 1 batch) at step 300: 0.0927
Seen so far: 9632 samples
Training loss (for 1 batch) at step 400: 0.3530
Seen so far: 12832 samples
Training loss (for 1 batch) at step 500: 0.3842
Seen so far: 16032 samples
Training loss (for 1 batch) at step 600: 0.6423
Seen so far: 19232 samples
Training loss (for 1 batch) at step 700: 0.1751
Seen so far: 22432 samples
Training loss (for 1 batch) at step 800: 0.4769
Seen so far: 25632 samples
Training loss (for 1 batch) at step 900: 0.1854
Seen so far: 28832 samples
Training loss (for 1 batch) at step 1000: 0.3130
Seen so far: 32032 samples
Training loss (for 1 batch) at step 1100: 0.1633
Seen so far: 35232 samples
Training loss (for 1 batch) at step 1200: 0.1446
Seen so far: 38432 samples
Training loss (for 1 batch) at step 1300: 0.4661
Seen so far: 41632 samples
Training loss (for 1 batch) at step 1400: 0.9977
Seen so far: 44832 samples
Training loss (for 1 batch) at step 1500: 0.3392
Seen so far: 48032 samples
Training acc over epoch: 0.9182
Validation acc: 0.9200
```

{{% /details %}}

이것으로 끝입니다!
