---
title: 함수형 API
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2019/03/01  
**{{< t f_last_modified >}}** 2023/06/25  
**{{< t f_description >}}** 함수형 API에 대한 완벽 가이드.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/functional_api.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/functional_api.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 셋업 {#setup}

```python
import numpy as np
import keras
from keras import layers
from keras import ops
```

## 소개 {#introduction}

Keras _함수형 API_ 는 [`keras.Sequential`]({{< relref "/docs/api/models/sequential#sequential-class" >}}) API보다 더 유연한 모델을 만드는 방법입니다.
함수형 API는 비선형 토폴로지, 공유 레이어, 심지어 여러 입력이나 출력이 있는 모델을 다룰 수 있습니다.

주요 아이디어는 딥러닝 모델이 일반적으로 레이어의 방향성이 있는 비순환 그래프(DAG, directed acyclic graph)라는 것입니다.
따라서 함수형 API는 _레이어 그래프_ 를 빌드하는 방법입니다.

### 함수형 API 모델 {#functional-api-model}

다음 모델을 생각해봅시다.

```mermaid
flowchart TD
    N1["input
    784-dimensional vectors"] --> N2["Dense
    (64 units, relu activation)"]
    N2 --> N3["Dense
    (64 units, relu activation)"]
    N3 --> N4["Dense
    (10 units, softmax activation)"]
    N4 --> N5["output
    logits of a probability distribution over 10 classes"]

    style N2 fill:#FFD600
    style N3 fill:#FFD600
    style N4 fill:#FFD600
```

이것은 3개의 레이어(노란색 `Dense` 레이어)가 있는 기본 그래프입니다.
함수형 API를 사용하여 이 모델을 빌드하려면, 먼저 입력 노드를 만듭니다.

```python
inputs = keras.Input(shape=(784,))
```

데이터 모양은 784차원 벡터로 설정됩니다.
각 샘플의 모양만 지정되므로, 배치 크기는 항상 생략됩니다.

예를 들어, `(32, 32, 3)` 모양의 이미지 입력이 있는 경우, 다음을 사용합니다.

```
# 단지 데모 목적입니다.
img_inputs = keras.Input(shape=(32, 32, 3))
```

반환된 `inputs`에는 모델에 공급하는 입력 데이터의 모양과 `dtype`에 대한 정보가 들어 있습니다.
모양은 다음과 같습니다.

```python
inputs.shape
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
(None, 784)
```

{{% /details %}}

dtype은 다음과 같습니다.

```python
inputs.dtype
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
'float32'
```

{{% /details %}}

이 `inputs` 객체에서 레이어를 호출하여, 레이어 그래프에 새 노드를 만듭니다.

```python
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
```

"레이어 호출" 작업은 "inputs"에서 생성한 이 레이어로 화살표를 그리는 것과 같습니다.
입력을 `dense` 레이어로 "전달"하고, 출력으로 `x`를 얻습니다.

레이어 그래프에 몇 개의 레이어를 더 추가해 보겠습니다.

```python
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
```

이 시점에서, 그래프 레이어에 입력과 출력을 지정하여 `Model`을 생성할 수 있습니다.

```python
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```

### 모델 요약 & 플롯 {#model-summary-and-plot}

모델 요약이 어떤지 살펴보겠습니다.

```python
import keras
from keras import layers

inputs = keras.Input(shape=(784,))
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```

```python
model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "mnist_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer_5 (InputLayer)           │ (None, 784)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_10 (Dense)                     │ (None, 64)                  │          50,240 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_11 (Dense)                     │ (None, 64)                  │           4,160 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_12 (Dense)                     │ (None, 10)                  │             650 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 55,050 (215.04 KB)
 Trainable params: 55,050 (215.04 KB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

모델을 그래프로 표시할 수도 있습니다.

```python
keras.utils.plot_model(model, "my_first_model.png")
```

![png](/images/guides/functional_api/model_plot.png)

그리고, 선택적으로, 플롯된 그래프에 각 레이어의 입력 및 출력 모양을 표시합니다.

```python
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
```

![png](/images/guides/functional_api/model_plot-2.png)

이 그림과 코드는 거의 동일합니다. 코드 버전에서는, 연결 화살표가 호출 연산으로 대체되었습니다.

"레이어 그래프"는 딥러닝 모델에 대한 직관적인 정신적 이미지이며,
함수형 API는 이를 밀접하게 반영하는 모델을 만드는 방법입니다.

## 트레이닝, 평가 및 추론 {#training-evaluation-and-inference}

트레이닝, 평가 및 추론은 함수형 API를 사용하여 빌드된 모델에서,
`Sequential` 모델과 정확히 동일한 방식으로 작동합니다.

`Model` 클래스는 빌트인 트레이닝 루프(`fit()` 메서드)와 빌트인 평가 루프(`evaluate()` 메서드)를 제공합니다.
이러한 루프를 쉽게 커스터마이즈하여, 당신만의 트레이닝 루틴을 구현할 수 있습니다.
`fit()`에서 발생하는 작업을 커스터마이즈하는 방법에 대한 가이드도 참조하세요.

- [TensorFlow로 커스텀 트레이닝 단계 작성]({{< relref "/docs/guides/custom_train_step_in_tensorflow" >}})
- [JAX로 커스텀 트레이닝 단계 작성]({{< relref "/docs/guides/custom_train_step_in_jax" >}})
- [PyTorch로 커스텀 트레이닝 단계 작성]({{< relref "/docs/guides/custom_train_step_in_torch" >}})

여기서, MNIST 이미지 데이터를 로드하고, 벡터로 reshape하고,
(검증 분할에서 성능을 모니터링하면서) 데이터에 대해 fit한 다음,
테스트 데이터에 대해 모델을 평가합니다.

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/2
 750/750 ━━━━━━━━━━━━━━━━━━━━ 1s 863us/step - accuracy: 0.8425 - loss: 0.5733 - val_accuracy: 0.9496 - val_loss: 0.1711
Epoch 2/2
 750/750 ━━━━━━━━━━━━━━━━━━━━ 1s 859us/step - accuracy: 0.9509 - loss: 0.1641 - val_accuracy: 0.9578 - val_loss: 0.1396
313/313 - 0s - 341us/step - accuracy: 0.9613 - loss: 0.1288
Test loss: 0.12876172363758087
Test accuracy: 0.9613000154495239
```

{{% /details %}}

자세한 내용은 [트레이닝 및 평가]({{< relref "/docs/guides/training_with_built_in_methods" >}}) 가이드를 참조하세요.

## 저장 및 직렬화 {#save-and-serialize}

모델 저장 및 직렬화는 `Sequential` 모델과 마찬가지로,
함수형 API를 사용하여 빌드한 모델에 대해서도 동일한 방식으로 작동합니다.
함수형 모델을 저장하는 표준 방법은 `model.save()`를 호출하여,
전체 모델을 단일 파일로 저장하는 것입니다.
심지어 나중에 모델을 빌드한 코드를 더 이상 사용할 수 없더라도,
이 파일에서 동일한 모델을 다시 만들 수 있습니다.

이 저장된 파일에는 다음이 포함됩니다.

- 모델 아키텍처
- 모델 가중치 값(트레이닝 중에 학습한 값)
- 모델 트레이닝 구성(있는 경우, `compile()`에 전달됨)
- 옵티마이저 및 해당 상태(있는 경우, 중단한 지점에서 트레이닝을 다시 시작하기 위함)

```python
model.save("my_model.keras")
del model
# 파일에서 정확히 동일한 모델을 재생성합니다.
model = keras.models.load_model("my_model.keras")
```

자세한 내용은 모델 [직렬화 및 저장]({{< relref "/docs/guides/serialization_and_saving" >}}) 가이드를 읽어보세요.

## 동일한 레이어 그래프를 사용하여 여러 모델 정의 {#use-the-same-graph-of-layers-to-define-multiple-models}

함수형 API에서, 모델은 레이어 그래프에서 입력과 출력을 지정하여 생성됩니다.
즉, 레이어의 단일 그래프를 사용하여 여러 모델을 생성할 수 있습니다.

아래 예에서, 동일한 레이어 스택을 사용하여 두 모델을 인스턴스화합니다.

- (1) 이미지 입력을 16차원 벡터로 변환하는 `encoder` 모델과
- (2) 트레이닝을 위한 엔드투엔드 `autoencoder` 모델입니다.

```python
encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "encoder"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ img (InputLayer)                │ (None, 28, 28, 1)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 26, 26, 16)     │           160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 24, 24, 32)     │         4,640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 8, 8, 32)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 6, 6, 32)       │         9,248 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 4, 4, 16)       │         4,624 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_max_pooling2d            │ (None, 16)             │             0 │
│ (GlobalMaxPooling2D)            │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 18,672 (72.94 KB)
 Trainable params: 18,672 (72.94 KB)
 Non-trainable params: 0 (0.00 B)
Model: "autoencoder"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ img (InputLayer)                │ (None, 28, 28, 1)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 26, 26, 16)     │           160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 24, 24, 32)     │         4,640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 8, 8, 32)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 6, 6, 32)       │         9,248 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 4, 4, 16)       │         4,624 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_max_pooling2d            │ (None, 16)             │             0 │
│ (GlobalMaxPooling2D)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ reshape (Reshape)               │ (None, 4, 4, 1)        │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose                │ (None, 6, 6, 16)       │           160 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_1              │ (None, 8, 8, 32)       │         4,640 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ up_sampling2d (UpSampling2D)    │ (None, 24, 24, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_2              │ (None, 26, 26, 16)     │         4,624 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_3              │ (None, 28, 28, 1)      │           145 │
│ (Conv2DTranspose)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 28,241 (110.32 KB)
 Trainable params: 28,241 (110.32 KB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

여기서, 디코딩 아키텍처는 인코딩 아키텍처와 엄격히 대칭적이므로,
출력 모양은 입력 모양 `(28, 28, 1)`과 동일합니다.

`Conv2D` 레이어의 역은 `Conv2DTranspose` 레이어이고,
`MaxPooling2D` 레이어의 역은 `UpSampling2D` 레이어입니다.

## 모든 모델은 (레이어와 마찬가지로) 호출 가능합니다. {#all-models-are-callable-just-like-layers}

`Input` 또는 다른 레이어의 출력에서 ​​호출하여, 모든 모델을 레이어인 것처럼 취급할 수 있습니다.
모델을 호출하면 모델의 아키텍처를 재사용하는 것뿐만 아니라, 가중치도 재사용하는 것입니다.

이를 실제로 보려면, 인코더 모델, 디코더 모델을 만들고, 두 호출로 체인하여,
오토인코더 모델을 얻는 오토인코더 예제에 대한 다른 방법이 있습니다.

```python
encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "encoder"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ original_img (InputLayer)       │ (None, 28, 28, 1)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (Conv2D)               │ (None, 26, 26, 16)     │           160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_5 (Conv2D)               │ (None, 24, 24, 32)     │         4,640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 8, 8, 32)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_6 (Conv2D)               │ (None, 6, 6, 32)       │         9,248 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_7 (Conv2D)               │ (None, 4, 4, 16)       │         4,624 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_max_pooling2d_1          │ (None, 16)             │             0 │
│ (GlobalMaxPooling2D)            │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 18,672 (72.94 KB)
 Trainable params: 18,672 (72.94 KB)
 Non-trainable params: 0 (0.00 B)
Model: "decoder"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ encoded_img (InputLayer)        │ (None, 16)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ reshape_1 (Reshape)             │ (None, 4, 4, 1)        │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_4              │ (None, 6, 6, 16)       │           160 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_5              │ (None, 8, 8, 32)       │         4,640 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ up_sampling2d_1 (UpSampling2D)  │ (None, 24, 24, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_6              │ (None, 26, 26, 16)     │         4,624 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_7              │ (None, 28, 28, 1)      │           145 │
│ (Conv2DTranspose)               │                        │               │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 9,569 (37.38 KB)
 Trainable params: 9,569 (37.38 KB)
 Non-trainable params: 0 (0.00 B)
Model: "autoencoder"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ img (InputLayer)                │ (None, 28, 28, 1)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ encoder (Functional)            │ (None, 16)             │        18,672 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ decoder (Functional)            │ (None, 28, 28, 1)      │         9,569 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 28,241 (110.32 KB)
 Trainable params: 28,241 (110.32 KB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

보시다시피, 모델은 중첩될 수 있습니다.
모델은 하위 모델을 포함할 수 있습니다. (모델은 레이어와 같기 때문입니다)
모델 중첩의 일반적인 사용 사례는 _앙상블_ 입니다.
예를 들어, 다음은 여러 모델을 예측의 평균을 내는 단일 모델로 앙상블하는 방법입니다.

```python
def get_model():
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)


model1 = get_model()
model2 = get_model()
model3 = get_model()

inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
```

## 복잡한 그래프 토폴로지 조작 {#manipulate-complex-graph-topologies}

### 여러 입력 및 출력이 있는 모델 {#models-with-multiple-inputs-and-outputs}

함수형 API를 사용하면, 여러 입력과 출력을 쉽게 조작할 수 있습니다.
이는 `Sequential` API로는 처리할 수 없습니다.

예를 들어, 우선순위에 따라 고객 문제 티켓을 순위를 매기고,
올바른 부서로 라우팅하는 시스템을 구축하는 경우, 모델에는 세 가지 입력이 있습니다.

- 티켓 제목 (텍스트 입력),
- 티켓 텍스트 본문 (텍스트 입력),
- 사용자가 추가한 어떤 태그들 (카테고리형 입력)

이 모델에는 두 가지 출력이 있습니다.

- 0~1 사이의 우선순위 점수(스칼라 sigmoid 출력),
- 티켓을 처리해야 하는 부서(부서 집합에 대한 softmax 출력).

함수형 API를 사용하면 몇 줄로 이 모델을 빌드할 수 있습니다.

```python
num_tags = 12  # 고유한 이슈 태그 수
num_words = 10000  # 텍스트 데이터를 전처리할 때, 얻은 어휘의 크기
num_departments = 4  # 예측을 위한 부서 수

title_input = keras.Input(
    shape=(None,), name="title"
)  # Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
tags_input = keras.Input(
    shape=(num_tags,), name="tags"
)  # Binary vectors of size `num_tags`

# Embed each word in the title into a 64-dimensional vector
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)

# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features, tags_input])

# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, name="priority")(x)
# Stick a department classifier on top of the features
department_pred = layers.Dense(num_departments, name="department")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs={"priority": priority_pred, "department": department_pred},
)
```

Now plot the model:

```python
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
```

![png](/images/guides/functional_api/functional_api_40_0.png)

When compiling this model, you can assign different losses to each output. You can even assign different weights to each loss – to modulate their contribution to the total training loss.

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[
        keras.losses.BinaryCrossentropy(from_logits=True),
        keras.losses.CategoricalCrossentropy(from_logits=True),
    ],
    loss_weights=[1.0, 0.2],
)
```

Since the output layers have different names, you could also specify the losses and loss weights with the corresponding layer names:

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "priority": keras.losses.BinaryCrossentropy(from_logits=True),
        "department": keras.losses.CategoricalCrossentropy(from_logits=True),
    },
    loss_weights={"priority": 1.0, "department": 0.2},
)
```

Train the model by passing lists of NumPy arrays of inputs and targets:

```python
# Dummy input data
title_data = np.random.randint(num_words, size=(1280, 12))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

# Dummy target data
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

model.fit(
    {"title": title_data, "body": body_data, "tags": tags_data},
    {"priority": priority_targets, "department": dept_targets},
    epochs=2,
    batch_size=32,
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/2
 40/40 ━━━━━━━━━━━━━━━━━━━━ 3s 57ms/step - loss: 1108.3792
Epoch 2/2
 40/40 ━━━━━━━━━━━━━━━━━━━━ 2s 54ms/step - loss: 621.3049

<keras.src.callbacks.history.History at 0x34afc3d90>
```

{{% /details %}}

When calling fit with a `Dataset` object, it should yield either a tuple of lists like `([title_data, body_data, tags_data], [priority_targets, dept_targets])` or a tuple of dictionaries like `({'title': title_data, 'body': body_data, 'tags': tags_data}, {'priority': priority_targets, 'department': dept_targets})`.

For more detailed explanation, refer to the [training and evaluation]({{< relref "/docs/guides/training_with_built_in_methods" >}}) guide.

### A toy ResNet model {#a-toy-resnet-model}

In addition to models with multiple inputs and outputs, the functional API makes it easy to manipulate non-linear connectivity topologies – these are models with layers that are not connected sequentially, which the `Sequential` API cannot handle.

A common use case for this is residual connections. Let's build a toy ResNet model for CIFAR10 to demonstrate this:

```python
inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "toy_resnet"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ img (InputLayer)    │ (None, 32, 32, 3) │          0 │ -                 │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_8 (Conv2D)   │ (None, 30, 30,    │        896 │ img[0][0]         │
│                     │ 32)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_9 (Conv2D)   │ (None, 28, 28,    │     18,496 │ conv2d_8[0][0]    │
│                     │ 64)               │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling2d_2     │ (None, 9, 9, 64)  │          0 │ conv2d_9[0][0]    │
│ (MaxPooling2D)      │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_10 (Conv2D)  │ (None, 9, 9, 64)  │     36,928 │ max_pooling2d_2[… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_11 (Conv2D)  │ (None, 9, 9, 64)  │     36,928 │ conv2d_10[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add (Add)           │ (None, 9, 9, 64)  │          0 │ conv2d_11[0][0],  │
│                     │                   │            │ max_pooling2d_2[… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_12 (Conv2D)  │ (None, 9, 9, 64)  │     36,928 │ add[0][0]         │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_13 (Conv2D)  │ (None, 9, 9, 64)  │     36,928 │ conv2d_12[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_1 (Add)         │ (None, 9, 9, 64)  │          0 │ conv2d_13[0][0],  │
│                     │                   │            │ add[0][0]         │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ conv2d_14 (Conv2D)  │ (None, 7, 7, 64)  │     36,928 │ add_1[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (None, 64)        │          0 │ conv2d_14[0][0]   │
│ (GlobalAveragePool… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_6 (Dense)     │ (None, 256)       │     16,640 │ global_average_p… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout (Dropout)   │ (None, 256)       │          0 │ dense_6[0][0]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_7 (Dense)     │ (None, 10)        │      2,570 │ dropout[0][0]     │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 223,242 (872.04 KB)
 Trainable params: 223,242 (872.04 KB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

Plot the model:

```python
keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)
```

![png](/images/guides/functional_api/functional_api_51_0.png)

Now train the model:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["acc"],
)
# We restrict the data to the first 1000 samples so as to limit execution time
# on Colab. Try to train on the entire dataset until convergence!
model.fit(
    x_train[:1000],
    y_train[:1000],
    batch_size=64,
    epochs=1,
    validation_split=0.2,
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 13/13 ━━━━━━━━━━━━━━━━━━━━ 1s 60ms/step - acc: 0.1096 - loss: 2.3053 - val_acc: 0.1150 - val_loss: 2.2973

<keras.src.callbacks.history.History at 0x1758bed40>
```

{{% /details %}}

## Shared layers {#shared-layers}

Another good use for the functional API are models that use _shared layers_. Shared layers are layer instances that are reused multiple times in the same model – they learn features that correspond to multiple paths in the graph-of-layers.

Shared layers are often used to encode inputs from similar spaces (say, two different pieces of text that feature similar vocabulary). They enable sharing of information across these different inputs, and they make it possible to train such a model on less data. If a given word is seen in one of the inputs, that will benefit the processing of all inputs that pass through the shared layer.

To share a layer in the functional API, call the same layer instance multiple times. For instance, here's an `Embedding` layer shared across two different text inputs:

```python
# Embedding for 1000 unique words mapped to 128-dimensional vectors
shared_embedding = layers.Embedding(1000, 128)

# Variable-length sequence of integers
text_input_a = keras.Input(shape=(None,), dtype="int32")

# Variable-length sequence of integers
text_input_b = keras.Input(shape=(None,), dtype="int32")

# Reuse the same layer to encode both inputs
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)
```

## Extract and reuse nodes in the graph of layers {#extract-and-reuse-nodes-in-the-graph-of-layers}

Because the graph of layers you are manipulating is a static data structure, it can be accessed and inspected. And this is how you are able to plot functional models as images.

This also means that you can access the activations of intermediate layers ("nodes" in the graph) and reuse them elsewhere – which is very useful for something like feature extraction.

Let's look at an example. This is a VGG19 model with weights pretrained on ImageNet:

```python
vgg19 = keras.applications.VGG19()
```

And these are the intermediate activations of the model, obtained by querying the graph data structure:

```python
features_list = [layer.output for layer in vgg19.layers]
```

Use these features to create a new feature-extraction model that returns the values of the intermediate layer activations:

```python
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)

img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)
```

This comes in handy for tasks like [neural style transfer]({{< relref "/docs/examples/generative/neural_style_transfer" >}}), among other things.

## Extend the API using custom layers {#extend-the-api-using-custom-layers}

`keras` includes a wide range of built-in layers, for example:

- Convolutional layers: `Conv1D`, `Conv2D`, `Conv3D`, `Conv2DTranspose`
- Pooling layers: `MaxPooling1D`, `MaxPooling2D`, `MaxPooling3D`, `AveragePooling1D`
- RNN layers: `GRU`, `LSTM`, `ConvLSTM2D`
- `BatchNormalization`, `Dropout`, `Embedding`, etc.

But if you don't find what you need, it's easy to extend the API by creating your own layers. All layers subclass the `Layer` class and implement:

- `call` method, that specifies the computation done by the layer.
- `build` method, that creates the weights of the layer (this is just a style convention since you can create weights in `__init__`, as well).

To learn more about creating layers from scratch, read [custom layers and models]({{< relref "/docs/guides/making_new_layers_and_models_via_subclassing" >}}) guide.

The following is a basic implementation of [`keras.layers.Dense`]({{< relref "/docs/api/layers/core_layers/dense#dense-class" >}}):

```python
class CustomDense(layers.Layer):
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


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)
```

For serialization support in your custom layer, define a `get_config()` method that returns the constructor arguments of the layer instance:

```python
class CustomDense(layers.Layer):
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


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)
config = model.get_config()

new_model = keras.Model.from_config(config, custom_objects={"CustomDense": CustomDense})
```

Optionally, implement the class method `from_config(cls, config)` which is used when recreating a layer instance given its config dictionary. The default implementation of `from_config` is:

```python
def from_config(cls, config):
  return cls(**config)
```

## When to use the functional API {#when-to-use-the-functional-api}

Should you use the Keras functional API to create a new model, or just subclass the `Model` class directly? In general, the functional API is higher-level, easier and safer, and has a number of features that subclassed models do not support.

However, model subclassing provides greater flexibility when building models that are not easily expressible as directed acyclic graphs of layers. For example, you could not implement a Tree-RNN with the functional API and would have to subclass `Model` directly.

For an in-depth look at the differences between the functional API and model subclassing, read [What are Symbolic and Imperative APIs in TensorFlow 2.0?](https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html).

### Functional API strengths: {#functional-api-strengths}

The following properties are also true for Sequential models (which are also data structures), but are not true for subclassed models (which are Python bytecode, not data structures).

#### Less verbose {#less-verbose}

There is no `super().__init__(...)`, no `def call(self, ...):`, etc.

Compare:

```python
inputs = keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
mlp = keras.Model(inputs, outputs)
```

With the subclassed version:

```python
class MLP(keras.Model):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.dense_1 = layers.Dense(64, activation='relu')
    self.dense_2 = layers.Dense(10)

  def call(self, inputs):
    x = self.dense_1(inputs)
    return self.dense_2(x)

# Instantiate the model.
mlp = MLP()
# Necessary to create the model's state.
# The model doesn't have a state until it's called at least once.
_ = mlp(ops.zeros((1, 32)))
```

#### Model validation while defining its connectivity graph {#model-validation-while-defining-its-connectivity-graph}

In the functional API, the input specification (shape and dtype) is created in advance (using `Input`). Every time you call a layer, the layer checks that the specification passed to it matches its assumptions, and it will raise a helpful error message if not.

This guarantees that any model you can build with the functional API will run. All debugging – other than convergence-related debugging – happens statically during the model construction and not at execution time. This is similar to type checking in a compiler.

#### A functional model is plottable and inspectable {#a-functional-model-is-plottable-and-inspectable}

You can plot the model as a graph, and you can easily access intermediate nodes in this graph. For example, to extract and reuse the activations of intermediate layers (as seen in a previous example):

```python
features_list = [layer.output for layer in vgg19.layers]
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)
```

#### A functional model can be serialized or cloned {#a-functional-model-can-be-serialized-or-cloned}

Because a functional model is a data structure rather than a piece of code, it is safely serializable and can be saved as a single file that allows you to recreate the exact same model without having access to any of the original code. See the [serialization & saving guide]({{< relref "/docs/guides/serialization_and_saving" >}}).

To serialize a subclassed model, it is necessary for the implementer to specify a `get_config()` and `from_config()` method at the model level.

### Functional API weakness: {#functional-api-weakness}

#### It does not support dynamic architectures {#it-does-not-support-dynamic-architectures}

The functional API treats models as DAGs of layers. This is true for most deep learning architectures, but not all – for example, recursive networks or Tree RNNs do not follow this assumption and cannot be implemented in the functional API.

## Mix-and-match API styles {#mix-and-match-api-styles}

Choosing between the functional API or Model subclassing isn't a binary decision that restricts you into one category of models. All models in the `keras` API can interact with each other, whether they're `Sequential` models, functional models, or subclassed models that are written from scratch.

You can always use a functional model or `Sequential` model as part of a subclassed model or layer:

```python
units = 32
timesteps = 10
input_dim = 5

# Define a Functional model
inputs = keras.Input((None, units))
x = layers.GlobalAveragePooling1D()(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)


class CustomRNN(layers.Layer):
    def __init__(self):
        super().__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        # Our previously-defined Functional model
        self.classifier = model

    def call(self, inputs):
        outputs = []
        state = ops.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = ops.stack(outputs, axis=1)
        print(features.shape)
        return self.classifier(features)


rnn_model = CustomRNN()
_ = rnn_model(ops.zeros((1, timesteps, input_dim)))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
(1, 10, 32)
(1, 10, 32)
```

{{% /details %}}

You can use any subclassed layer or model in the functional API as long as it implements a `call` method that follows one of the following patterns:

- `call(self, inputs, **kwargs)` – Where `inputs` is a tensor or a nested structure of tensors (e.g. a list of tensors), and where `**kwargs` are non-tensor arguments (non-inputs).
- `call(self, inputs, training=None, **kwargs)` – Where `training` is a boolean indicating whether the layer should behave in training mode and inference mode.
- `call(self, inputs, mask=None, **kwargs)` – Where `mask` is a boolean mask tensor (useful for RNNs, for instance).
- `call(self, inputs, training=None, mask=None, **kwargs)` – Of course, you can have both masking and training-specific behavior at the same time.

Additionally, if you implement the `get_config` method on your custom Layer or model, the functional models you create will still be serializable and cloneable.

Here's a quick example of a custom RNN, written from scratch, being used in a functional model:

```python
units = 32
timesteps = 10
input_dim = 5
batch_size = 16


class CustomRNN(layers.Layer):
    def __init__(self):
        super().__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        self.classifier = layers.Dense(1)

    def call(self, inputs):
        outputs = []
        state = ops.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = ops.stack(outputs, axis=1)
        return self.classifier(features)


# Note that you specify a static batch size for the inputs with the `batch_shape`
# arg, because the inner computation of `CustomRNN` requires a static batch size
# (when you create the `state` zeros tensor).
inputs = keras.Input(batch_shape=(batch_size, timesteps, input_dim))
x = layers.Conv1D(32, 3)(inputs)
outputs = CustomRNN()(x)

model = keras.Model(inputs, outputs)

rnn_model = CustomRNN()
_ = rnn_model(ops.zeros((1, 10, 5)))
```
