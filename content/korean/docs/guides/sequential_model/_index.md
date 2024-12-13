---
title: Sequential 모델
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2020/04/12  
**{{< t f_last_modified >}}** 2023/06/25  
**{{< t f_description >}}** Sequential 모델에 대한 완벽 가이드.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/sequential_model.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/sequential_model.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 셋업 {#setup}

```python
import keras
from keras import layers
from keras import ops
```

## Sequential 모델을 사용하는 경우 {#when-to-use-a-sequential-model}

`Sequential` 모델은 각 레이어에
**정확히 하나의 입력 텐서와 하나의 출력 텐서**가 있는
**레이어들의 일반 스택**에 적합합니다.

개략적으로, 다음의 `Sequential` 모델을 봅시다.

```python
# 3개의 레이어로 Sequential 모델 정의
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# 테스트 입력에 대해 모델 호출
x = ops.ones((3, 3))
y = model(x)
```

이 함수형 모델과 동일합니다:

```python
# 3개의 레이어를 만듭니다
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# 테스트 입력에 대해 레이어 호출
x = ops.ones((3, 3))
y = layer3(layer2(layer1(x)))
```

Sequential 모델은 다음과 같은 경우, **적절하지 않습니다**.

- 모델에 여러 입력 또는 여러 출력이 있는 경우
- 레이어들 중 어떤 것에 여러 입력 또는 여러 출력이 있는 경우
- 레이어 공유가 필요한 경우
- 비선형 토폴로지(예: residual 연결, 다중 분기 모델)가 필요한 경우

## Sequential 모델 생성 {#creating-a-sequential-model}

Sequential 생성자에 레이어 리스트를 전달하여 Sequential 모델을 생성할 수 있습니다.

```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)
```

레이어는 `layers` 속성을 통해 접근할 수 있습니다.

```python
model.layers
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
[<Dense name=dense, built=False>,
 <Dense name=dense_1, built=False>,
 <Dense name=dense_2, built=False>]
```

{{% /details %}}

`add()` 메서드를 통해 Sequential 모델을 증분 방식(incrementally)으로 생성할 수도 있습니다.

```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```

레이어를 제거하는 `pop()` 메서드도 있습니다.
Sequential 모델은 레이어 리스트와 매우 비슷하게 동작합니다.

```python
model.pop()
print(len(model.layers))  # 2
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
2
```

{{% /details %}}

또한 Sequential 생성자는 (Keras의 모든 레이어나 모델과 마찬가지로) `name` 인수를 허용합니다.
이는 의미적으로(semantically) 의미 있는(meaningful) 이름으로 TensorBoard 그래프에 어노테이션 하기에 유용합니다.

```python
model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4, name="layer3"))
```

## 입력 모양을 미리 지정 {#specifying-the-input-shape-in-advance}

일반적으로, Keras의 모든 레이어는 가중치를 생성하기 위해 입력의 모양을 알아야 합니다.
따라서 이와 같은 레이어를 만들면, 처음에는 가중치가 없습니다.

```python
layer = layers.Dense(3)
layer.weights  # 비어 있음
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
[]
```

{{% /details %}}

입력에 대해 처음 호출될 때 가중치를 생성합니다.
가중치의 모양은 입력의 모양에 따라 달라지기 때문입니다.

```python
# 테스트 입력에 대한 레이어 호출
x = ops.ones((1, 4))
y = layer(x)
layer.weights  # 이제 (4, 3) 및 (3,) 모양의 가중치가 있습니다.
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
[<KerasVariable shape=(4, 3), dtype=float32, path=dense_6/kernel>,
 <KerasVariable shape=(3,), dtype=float32, path=dense_6/bias>]
```

{{% /details %}}

당연히, 이는 Sequential 모델에도 적용됩니다.
입력 모양 없이 Sequential 모델을 인스턴스화하면, "빌드"되지 않습니다.
가중치가 없습니다. (그리고 `model.weights`를 호출하면 이 오류만 발생합니다)
가중치는 모델이 처음 입력 데이터를 볼 때 생성됩니다.

```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)  # 이 단계에서는 가중치가 없습니다!

# 이 시점에서는, 다음을 수행할 수 없습니다.
# model.weights

# 다음 역시도 할 수 없습니다.
# model.summary()

# 테스트 입력에 대해 모델 호출
x = ops.ones((1, 4))
y = model(x)
print("Number of weights after calling the model:", len(model.weights))  # 6
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Number of weights after calling the model: 6
```

{{% /details %}}

모델이 "빌드"되면, `summary()` 메서드를 호출하여 내용을 표시할 수 있습니다.

```python
model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "sequential_3"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ dense_7 (Dense)                 │ (1, 2)                    │         10 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_8 (Dense)                 │ (1, 3)                    │          9 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_9 (Dense)                 │ (1, 4)                    │         16 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 35 (140.00 B)
 Trainable params: 35 (140.00 B)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

그러나, Sequential 모델을 점진적으로(incrementally) 빌드할 때,
지금까지의 모델 요약을 현재 출력 모양을 포함하여 표시할 수 있다면 매우 유용할 수 있습니다.
이 경우, `Input` 객체를 모델에 전달하여 모델을 시작해야 하며, 이렇게 하면 처음부터 입력 모양을 알 수 있습니다.

```python
model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2, activation="relu"))

model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "sequential_4"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ dense_10 (Dense)                │ (None, 2)                 │         10 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 10 (40.00 B)
 Trainable params: 10 (40.00 B)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

`Input` 객체는 레이어가 아니기 때문에, `model.layers`의 일부로 표시되지 않습니다.

```python
model.layers
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
[<Dense name=dense_10, built=True>]
```

{{% /details %}}

이와 같이 미리 정의된 입력 모양으로 빌드된 모델은,
항상 가중치를 갖고(데이터를 보기 전에도) 항상 정의된 출력 모양을 갖습니다.

일반적으로, Sequential 모델의 입력 모양을 알고 있다면,
항상 미리 지정하는 것이 권장되는 모범 사례입니다.

## 일반적인 디버깅 워크플로: `add()` + `summary()` {#a-common-debugging-workflow-add-summary}

새로운 Sequential 아키텍처를 빌드할 때,
`add()`로 레이어를 점진적으로 쌓고(incrementally stack) 모델 요약을 자주 출력하는 것이 유용합니다.
예를 들어, 이를 통해 `Conv2D` 및 `MaxPooling2D` 레이어의 스택이,
이미지 특성 맵을 다운샘플링하는 방식을 모니터링할 수 있습니다.

```python
model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB 이미지
model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))

# 이 시점에서 현재 출력 모양이 무엇인지 추측할 수 있나요? 아마도 아닐 겁니다.
# 그냥 출력해 봅시다:
model.summary()

# 답은 (40, 40, 32)입니다. 따라서 계속해서 다운샘플링할 수 있습니다.

model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

# 그리고 지금은?
model.summary()

# 이제 4x4 특성 맵이 있으므로, 글로벌 최대 풀링을 적용할 시간입니다.
model.add(layers.GlobalMaxPooling2D())

# 마지막으로, 분류 레이어를 추가합니다.
model.add(layers.Dense(10))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "sequential_5"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 123, 123, 32)      │      2,432 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (Conv2D)               │ (None, 121, 121, 32)      │      9,248 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 40, 40, 32)        │          0 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 11,680 (45.62 KB)
 Trainable params: 11,680 (45.62 KB)
 Non-trainable params: 0 (0.00 B)
Model: "sequential_5"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 123, 123, 32)      │      2,432 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (Conv2D)               │ (None, 121, 121, 32)      │      9,248 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 40, 40, 32)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_2 (Conv2D)               │ (None, 38, 38, 32)        │      9,248 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_3 (Conv2D)               │ (None, 36, 36, 32)        │      9,248 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 12, 12, 32)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_4 (Conv2D)               │ (None, 10, 10, 32)        │      9,248 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_5 (Conv2D)               │ (None, 8, 8, 32)          │      9,248 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 4, 4, 32)          │          0 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 48,672 (190.12 KB)
 Trainable params: 48,672 (190.12 KB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

매우 실용적이죠?

## 모델이 생기면 무엇을 해야 하나요? {#what-to-do-once-you-have-a-model}

모델 아키텍처가 준비되면 다음을 수행해야 합니다.

- 모델을 트레이닝하고, 평가하고, 추론을 실행합니다.
  {{< titledRelref "/docs/guides/training_with_built_in_methods" >}} 가이드를 참조하세요.
- 모델을 디스크에 저장하고 복원합니다.
  {{< titledRelref "/docs/guides/serialization_and_saving" >}} 가이드를 참조하세요.

## Sequential 모델을 사용한 특성 추출 {#feature-extraction-with-a-sequential-model}

Sequential 모델이 빌드되면,
{{< titledRelref "/docs/guides/functional_api" >}} 모델처럼 동작합니다.
즉, 모든 레이어에 `input` 및 `output` 속성이 있습니다.
이러한 속성을 사용하면, Sequential 모델에서 모든 중간 레이어의 출력을 추출하는 모델을 빠르게 만드는 것과 같은,
멋진 작업을 수행할 수 있습니다.

```python
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# 테스트 입력에 대해 feature_extractor를 호출합니다.
x = ops.ones((1, 250, 250, 3))
features = feature_extractor(x)
```

한 레이어에서만 특성을 추출하는 유사한 예는 다음과 같습니다.

```python
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# 테스트 입력에 대해 feature_extractor를 호출합니다.
x = ops.ones((1, 250, 250, 3))
features = feature_extractor(x)
```

## Sequential 모델을 사용한 전이 학습 {#transfer-learning-with-a-sequential-model}

전이 학습은 모델의 하위 레이어를 동결하고, 상위 레이어만 트레이닝하는 것으로 구성됩니다.
익숙하지 않은 경우, {{< titledRelref "/docs/guides/transfer_learning" >}} 가이드를 읽어보세요.

다음은 Sequential 모델과 관련된 두 가지 일반적인 전이 학습 청사진입니다.

먼저, Sequential 모델이 있고, 마지막 레이어를 제외한 모든 레이어를 동결하려고 한다고 가정해 보겠습니다.
이 경우, `model.layers`에 걸쳐 반복하고,
마지막 레이어를 제외한 모든 레이어에 `layer.trainable = False`를 설정하면 됩니다.
다음과 같습니다.

```python
model = keras.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10),
])

# 아마도 먼저 사전 트레이닝된 가중치를 로드하고 싶을 것입니다.
model.load_weights(...)

# 마지막 레이어를 제외한 모든 레이어를 동결합니다.
for layer in model.layers[:-1]:
  layer.trainable = False

# 다시 컴파일하고 트레이닝시킵니다. (이렇게 하면 마지막 레이어의 가중치만 업데이트됩니다)
model.compile(...)
model.fit(...)
```

또다른 일반적인 청사진은 Sequential 모델을 사용하여,
사전 트레이닝된 모델과 새로 초기화된 분류 레이어를 쌓는 것입니다. 다음과 같습니다.

```python
# 사전 트레이닝된 가중치를 사용하여, 컨볼루션 베이스 불러오기
base_model = keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    pooling='avg')

# 베이스 모델 동결
base_model.trainable = False

# Sequential 모델을 사용하여, 상단에(on top) 트레이닝 가능한 분류기를 추가합니다.
model = keras.Sequential([
    base_model,
    layers.Dense(1000),
])

# 컴파일 & 트레이닝
model.compile(...)
model.fit(...)
```

전이 학습을 한다면, 아마도 이 두 패턴을 자주 사용하게 될 것입니다.

Sequential 모델에 대해 알아야 할 것은 이것뿐입니다!

Keras에서 모델을 빌드하는 방법에 대한 자세한 내용은, 다음을 참조하세요.

- {{< titledRelref "/docs/guides/functional_api" >}} 가이드
- {{< titledRelref "/docs/guides/making_new_layers_and_models_via_subclassing" >}} 가이드
