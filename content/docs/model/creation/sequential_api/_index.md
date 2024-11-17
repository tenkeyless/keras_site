---
title: Sequential API 방식
linkTitle: Sequential API
toc: true
weight: 1
type: docs
prev: docs/model/creation
next: docs/model/creation/functional_api
---

## Sequential 모델

Sequential 모델은 계층적으로 순차적인 구조를 가지며, 각 층(layer)이 순서대로 연결됩니다.

`keras.Sequential`은 모델이 순수하게 단일 입력, 단일 출력 레이어의 스택으로서, 모델의 특별한 경우입니다.

### `Sequential` 클래스

```python
keras.Sequential(layers=None, trainable=True, name=None)
```

`Sequential`은 선형 레이어 스택을 `Model`로 그룹화합니다.

## 예제

### 예제 1. Sequential 모델

```python
import keras

model = keras.Sequential()
model.add(keras.Input(shape=(16,)))
model.add(keras.layers.Dense(8))
```

이 모델의 입출력은 다음과 같습니다.

- 입력 : 크기가 16인 텐서
- 출력 : 크기가 8인 텐서

{{% details title="분석" closed="true" %}}

모델 요약 결과는 다음과 같습니다.

```python
model.summary()
```

```plain
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 8)                   │             136 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 136 (544.00 B)
 Trainable params: 136 (544.00 B)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

### 예제 2. 지연된 빌드 패턴

```python
import keras

model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(4))
```

초기 `Input`을 생략할 수도 있습니다.

이 경우, 모델은 트레이닝/평가 메소드를 처음 호출할 때까지 가중치를 갖지 않습니다. (아직 빌드되지 않았기 때문입니다)

따라서, 이 시점에서는 `model.weights`가 아직 생성되지 않았습니다.

{{% details title="분석" closed="true" %}}

모델 요약 결과는 다음과 같습니다.

```python
model.summary()
```

```plain
Model: "sequential_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense_1 (Dense)                      │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ ?                           │     0 (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 0 (0.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

### 예제 3. 입력에 대한 모델 빌드

```python
import keras

model = keras.Sequential()
model.add(keras.Input(shape=(16,)))
model.add(keras.layers.Dense(8))
len(model.weights)  # "2"를 리턴
```

반면, `Input`을 지정하면, 레이어를 추가할 때마다 모델이 계속해서 빌드됩니다.

`len(model.weights)`가 `2`를 리턴하는데, 이는 가중치가 있음을 의미합니다.

{{% details title="분석" closed="true" %}}

모델 요약 결과는 다음과 같습니다.

```python
model.summary()
```

```plain
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense_3 (Dense)                      │ (None, 8)                   │             136 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 136 (544.00 B)
 Trainable params: 136 (544.00 B)
 Non-trainable params: 0 (0.00 B)
```

모델의 가중치를 살펴봅시다.

```python
model.weights
```

```plain
[<KerasVariable shape=(16, 8), dtype=float32, path=sequential_2/dense_3/kernel>,
 <KerasVariable shape=(8,), dtype=float32, path=sequential_2/dense_3/bias>]
```

결과는 이와 같으며, 이는 Dense 레이어에 대해, kernel 가중치와 bias 가중치를 가짐을 의미합니다.

{{% /details %}}

### 예제 4. 지연된 빌드 패턴 1

```python
import keras

model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(4))
model.build((None, 16))
len(model.weights)  # "4"를 리턴
```

지연된 빌드 패턴(delayed-build pattern, 지정된 입력 형태 없음)을 사용하는 경우,
`build(batch_input_shape)`를 호출하여,
모델을 수동으로 빌드하도록 선택할 수 있습니다.

{{% details title="분석" closed="true" %}}

```python
import keras

model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(4))
```

여기까지만 수행하고, 가중치의 길이를 살펴보면,

```python
len(model.weights)
```

```plain
0
```

0이 출력됩니다. 가중치가 아직 존재하지 않습니다.

모델 요약을 살펴보면 다음과 같습니다.

```python
model.summary()
```

```plain
Model: "sequential_4"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense_6 (Dense)                      │ ?                           │     0 (unbuilt) │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_7 (Dense)                      │ ?                           │     0 (unbuilt) │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 0 (0.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 0 (0.00 B)
```

모델을 빌드해 봅시다.

```python
model.build((None, 16))
```

가중치의 길이를 살펴보면,

```python
len(model.weights)
```

```plain
4
```

4가 출력됩니다. 요약을 살펴보면 다음과 같습니다.

```python
model.summary()
```

```plain
Model: "sequential_4"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense_6 (Dense)                      │ (None, 8)                   │             136 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_7 (Dense)                      │ (None, 4)                   │              36 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 172 (688.00 B)
 Trainable params: 172 (688.00 B)
 Non-trainable params: 0 (0.00 B)
```

결과는 이와 같으며, Dense 레이어에 대해, kernel과 bias 2개를 가지기 때문에, 2개 레이어이므로, 가중치의 길이가 4가 되는 것입니다.

이 상태에서, 모델을 다시 빌드하려고 하면 에러가 발생합니다.

```python
model.build((None, 32))
```

```plain
...

ValueError: Sequential model 'sequential_4' has already been configured to use input shape (None, 16). You cannot build it with input_shape (None, 32)
```

이미 `(None, 16)`에 대해 빌드되어 모양이 고정되었기 때문에, 다시 새로운 것으로 바꾸어 빌드할 수 없음을 알 수 있습니다.

{{% /details %}}

### 예제 5. 지연된 빌드 패턴 2

```python
import keras

model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(1))
model.compile(optimizer='sgd', loss='mse')
# 이것은 처음으로 모델을 빌드합니다.
model.fit(x, y, batch_size=32, epochs=10)
```

지연된 빌드 패턴(지정된 입력 형태 없음)을 사용하는 경우,
`fit`, `eval` 또는 `predict`를 처음 호출하거나,
일부 입력 데이터에 대해 모델을 처음 호출할 때 모델이 빌드됩니다.

{{% details title="분석" closed="true" %}}

이 명령은 에러가 발생합니다.

왜냐하면, `x` 및 `y`가 정의되지 않았기 때문입니다.

다만, 위의 경우에 비추어 볼 때, 입력 모양인 `x`와 출력 모양인 `y`에 대해 모델이 맞추어 만들어지게 되고, 빌드가 됨을 알 수 있습니다.

그리고, 위의 예제 4를 토대로, 한 번 fit을 통해 모양이 고정되면, 변경하지 못함도 알 수 있습니다.

{{% /details %}}

### 예제 6. 이미지 모델

```python
import keras

# 모델 생성
model = keras.Sequential([
    keras.Input(shape=(None, None, 3)),
    keras.layers.Conv2D(filters=32, kernel_size=3),
])
```

이 모델의 입출력은 다음과 같습니다.

- 입력 : 가로, 세로 크기가 지정되지 않은 3차원 텐서
  - 일반적으로 이런 경우는 크기가 제한되지 않은 이미지입니다.
- 출력 : 가로, 세로 크기가 지정되지 않은 32차원 텐서

이런 경우는 아직 모델이 완성되지는 않았으나, 이미지를 가지고 처리하는 모델의 일종이라고 생각할 수 있습니다.

{{% details title="분석" closed="true" %}}

이 모델의 요약을 살펴보면 다음과 같습니다.

```python
model.summary()
```

```plain
Model: "sequential_6"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, None, None, 32)      │             896 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 896 (3.50 KB)
 Trainable params: 896 (3.50 KB)
 Non-trainable params: 0 (0.00 B)
```

모양은 처음에 입력에 대해 모양이 주어졌으므로, 모양이 고정된 모델이며, 가중치가 존재함을 알 수 있습니다.

{{% /details %}}
