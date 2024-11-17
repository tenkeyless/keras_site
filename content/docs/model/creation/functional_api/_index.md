---
title: Functional API 방식
linkTitle: Functional API
toc: true
weight: 2
type: docs
prev: docs/model/creation/sequential_api
---

## Functional 모델

임의의 모델 아키텍처를 지원하는, 사용하기 쉽고 모든 기능을 갖춘 API인 Functional API입니다.

대부분의 사람들과 대부분의 사용 사례에서 이것이 사용해야 하는 것입니다. 이것이 Keras의 "산업 강점" 모델입니다.

## 간단한 예제

Functional API는 복잡한 모델, 예를 들어 다중 입력 또는 출력을 가지는 모델, 비순차적 연결을 허용하는 모델 등을 만들 때 유용합니다.

1. `Input`에서 시작하여,
2. 레이어 호출을 연결하여 모델의 정방향 전달 단계를 지정하고,
3. 마지막으로 입력 및 출력으로부터 ​​모델을 만듭니다.

```python
import keras

# 입력층 정의
inputs = keras.Input(shape=(37,))

# 은닉층과 출력층 정의
x = keras.layers.Dense(32, activation="relu")(inputs)
outputs = keras.layers.Dense(5, activation="softmax")(x)

# 모델 생성
model = keras.Model(inputs=inputs, outputs=outputs)
```

참고: 입력 텐서의 dict, list 및 tuple만 지원됩니다. 중첩된 입력은 지원되지 않습니다. (예: 리스트의 리스트 또는 dict의 dict)

중간 텐서를 사용하여, 새로운 Functional API 모델을 생성할 수도 있습니다.

이를 통해 모델의 하위 구성요소를 빠르게 추출할 수 있습니다.

### 예제 1. 중간 텐서를 사용한 새로운 모델

```python
inputs = keras.Input(shape=(None, None, 3))
processed = keras.layers.RandomCrop(width=128, height=128)(inputs)
conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed)
pooling = keras.layers.GlobalAveragePooling2D()(conv)
feature = keras.layers.Dense(10)(pooling)

full_model = keras.Model(inputs, feature)
backbone = keras.Model(processed, conv)
activations = keras.Model(conv, feature)
```

`backbone` 및 `activations` 모델은 `keras.Input` 객체로 생성되지 않고,
`keras.Input` 객체에서 생성된 텐서로 생성됩니다.

내부적으로는, 레이어와 가중치가 이러한 모델 전체에서 공유되므로,
사용자는 `full_model`을 트레이닝하고, `backbone` 이나 `activations`를 사용하여 특성 추출을 수행할 수 있습니다.

모델의 입력과 출력은 텐서의 중첩 구조일 수도 있으며, 생성된 모델은 기존 API를 모두 지원하는 표준 Functional API 모델입니다.
