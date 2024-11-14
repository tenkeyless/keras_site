---
title: Sequential API 방식
toc: true
weight: 1
type: docs
prev: docs/model/creation
next: docs/model/creation/functional-api
---

## Sequential 모델

Sequential 모델은 계층적으로 순차적인 구조를 가지며, 각 층(layer)이 순서대로 연결됩니다.

`keras.Sequential`은 모델이 순수하게 단일 입력, 단일 출력 레이어의 스택으로서, 모델의 특별한 경우입니다.

### `Sequential` 클래스

```python
keras.Sequential(layers=None, trainable=True, name=None)
```

`Sequential`은 선형 레이어 스택을 `Model`로 그룹화합니다.

## 예시

### 예시 1. Sequential 모델

```python
import keras

model = keras.Sequential()
model.add(keras.Input(shape=(16,)))
model.add(keras.layers.Dense(8))
```

이 모델의 입출력은 다음과 같습니다.

- 입력 : 크기가 16인 텐서
- 출력 : 크기가 8인 텐서

### 예시 2. 지연된 빌드 패턴

```python
import keras

model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(4))
```

초기 `Input`을 생략할 수도 있습니다.

이 경우, 모델은 트레이닝/평가 메소드를 처음 호출할 때까지 가중치를 갖지 않습니다. (아직 빌드되지 않았기 때문입니다)

따라서, 이 시점에서는 `model.weights`가 아직 생성되지 않았습니다.

### 예시 3. 입력에 대한 모델 빌드

```python
import keras

model = keras.Sequential()
model.add(keras.Input(shape=(16,)))
model.add(keras.layers.Dense(8))
len(model.weights)  # "2"를 리턴
```

반면, `Input`을 지정하면, 레이어를 추가할 때마다 모델이 계속해서 빌드됩니다.

`len(model.weights)`가 `2`를 리턴하는데, 이는 가중치가 있음을 의미합니다.

### 예시 4. 지연된 빌드 패턴 1

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

### 예시 5. 지연된 빌드 패턴 2

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

### 예시 6. 이미지 모델

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
