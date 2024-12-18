---
title: EfficientNet으로 미세 조정을 통한 이미지 분류
linkTitle: EfficientNet 이미지 분류
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

**{{< t f_author >}}** [Yixing Fu](https://github.com/yixingfu)  
**{{< t f_date_created >}}** 2020/06/30  
**{{< t f_last_modified >}}** 2023/07/10  
**{{< t f_description >}}** Stanford Dogs 분류를 위해 imagenet에 대해 사전 트레이닝된 가중치로 EfficientNet을 사용합니다.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_efficientnet_fine_tuning.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_efficientnet_fine_tuning.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개: EfficientNet이란? {#introduction-what-is-efficientnet}

[Tan and Le, 2019](https://arxiv.org/abs/1905.11946)에서
처음 소개된 EfficientNet은 imagenet과 일반적인 이미지 분류 전이 학습 작업 모두에서
SOTA 정확도에 도달하는 가장 효율적인 모델(즉, 추론에 최소의 FLOPS가 필요함)중 하나입니다.

가장 작은 기본 모델은 훨씬 더 작은 모델로 SOTA에 가까운 정확도를 달성한
[MnasNet](https://arxiv.org/abs/1807.11626)과 유사합니다.
모델을 확장하는 휴리스틱 방식을 도입함으로써,
EfficientNet은 다양한 규모에서 효율성과 정확성의 좋은 조합을 나타내는 모델 제품군(B0~B7)을 제공합니다.
이러한 확장 휴리스틱(복합 확장, 자세한 내용은 [Tan and Le, 2019](https://arxiv.org/abs/1905.11946) 참조)을 사용하면
효율성 지향 기본 모델(B0)이 모든 규모의 모델을 능가하는 동시에,
하이퍼파라미터에 대한 광범위한 그리드 검색을 피할 수 있습니다.

이 모델의 최신 업데이트 요약은 [여기](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)에서 확인할 수 있으며,
모델의 이미지넷 성능을 더욱 향상시키기 위해 다양한 보강 체계와 준지도 학습 접근법이 적용됩니다.
이러한 모델 확장은 모델 아키텍처를 변경하지 않고 가중치를 업데이트하여 사용할 수 있습니다.

## B0~B7 EfficientNet 변형 {#b0-to-b7-variants-of-efficientnet}

_(이 섹션에서는 "복합 스케일링(compound scaling)"에 대한 자세한 내용을 제공하며, 모델 사용에만 관심이 있는 경우 건너뛸 수 있습니다)_

[원본 논문](https://arxiv.org/abs/1905.11946)을 보면 EfficientNet이
논문의 식 (3)과 같이 스케일링 계수를 임의로 선택해 만든 연속적인 모델군이라는 인상을 받을 수 있습니다.
그러나, 해상도, 깊이, 폭의 선택은 여러 가지 요인에 의해 제한됩니다:

- 해상도: 8, 16 등으로 나눌 수 없는 해상도는 일부 레이어의 경계 근처에서 제로 패딩이 발생하여
  계산 리소스를 낭비합니다. 이는 특히 모델의 작은 변형에 적용되므로,
  B0 및 B1의 입력 해상도는 224 및 240으로 선택됩니다.

- 깊이와 너비: EfficientNet의 빌딩 블록은 채널 크기가 8의 배수일 것을 요구합니다.

- 리소스 제한: 메모리 제한으로 인해 깊이와 너비가 계속 증가할 수 있는 경우
  해상도에 병목 현상이 발생할 수 있습니다.
  이러한 상황에서는, 깊이 및/또는 너비를 늘리되 해상도를 유지하면 성능이 향상될 수 있습니다.

따라서, EfficientNet 모델의 각 변형 모델의 깊이, 너비 및 해상도는 수작업으로 선택되어
좋은 결과를 내는 것으로 입증되었지만, 복합 스케일링 공식과 크게 다를 수 있습니다.
따라서, keras 구현(아래에 자세히 설명되어 있음)에서는
너비/깊이/해상도 매개변수의 임의 선택을 허용하는 대신,
이 8가지 모델인 B0~B7만 제공합니다.

## EfficientNet Keras 구현 {#keras-implementation-of-efficientnet}

Keras 2.3 버전부터 Keras와 함께 EfficientNet B0~B7의 구현이 제공되었습니다.
ImageNet으로부터의 1000개의 이미지 클래스를 분류하는데
EfficientNetB0을 사용하려면, 다음을 실행하세요:

```python
from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights='imagenet')
```

이 모델은 `(224, 224, 3)` 모양의 입력 이미지를 사용하며,
입력 데이터는 `[0, 255]` 범위여야 합니다.
정규화는 모델의 일부로 포함됩니다.

이미지넷에 대해 EfficientNet을 트레이닝하려면
엄청난 양의 리소스와 모델 아키텍처 자체에 포함되지 않은 몇 가지 기술이 필요하기 때문입니다.
따라서 Keras 구현은 기본적으로
[AutoAugment](https://arxiv.org/abs/1805.09501)을 통한 트레이닝을 통해 얻은
사전 트레이닝된 가중치를 로드합니다.

B0~B7 베이스 모델의 경우, 입력 모양이 다릅니다.
다음은 각 모델에 대해 기대되는 입력 모양 목록입니다:

| 베이스 모델    | 해상도 |
| -------------- | ------ |
| EfficientNetB0 | 224    |
| EfficientNetB1 | 240    |
| EfficientNetB2 | 260    |
| EfficientNetB3 | 300    |
| EfficientNetB4 | 380    |
| EfficientNetB5 | 456    |
| EfficientNetB6 | 528    |
| EfficientNetB7 | 600    |

모델이 전이 학습을 목적으로 하는 경우, Keras 구현은 최상위 레이어를 제거하는 옵션을 제공합니다:

```python
model = EfficientNetB0(include_top=False, weights='imagenet')
```

이 옵션은 두 번째 레이어의 1280개 특성을
1000개의 ImageNet 클래스에 대한 예측으로 바꾸는 최종 `Dense` 레이어를 제외합니다.
최상위 레이어를 커스텀 레이어로 대체하면
전이 학습 워크플로우에서 EfficientNet을 특성 추출기로 사용할 수 있습니다.

모델 생성자에서 주목할 만한 또 다른 인수는
[확률적 깊이(stochastic depth)](https://arxiv.org/abs/1603.09382)를
담당하는 드롭아웃 비율을 제어하는 `drop_connect_rate`입니다.
이 매개변수는 미세 조정에서 추가 정규화를 위한 토글 역할을 하지만,
로드된 가중치에는 영향을 미치지 않습니다.
예를 들어, 더 강력한 정규화가 필요한 경우, 다음을 사용해 보세요:

```python
model = EfficientNetB0(weights='imagenet', drop_connect_rate=0.4)
```

기본값은 0.2입니다.

## 예시: Stanford Dogs를 위한 EfficientNetB0 {#example-efficientnetb0-for-stanford-dogs}

EfficientNet은 광범위한 이미지 분류 작업을 수행할 수 있습니다.
따라서, 전이 학습을 위한 좋은 모델입니다.
엔드투엔드 예시로, [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) 데이터 세트에 대해
사전 트레이닝된 EfficientNetB0을 사용해 보여드리겠습니다.

## 셋업 및 데이터 로드 {#setup-and-data-loading}

```python
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf  # tf.data를 위해
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.applications import EfficientNetB0

# IMG_SIZE는 EfficientNet 모델 선택에 따라 결정됩니다.
IMG_SIZE = 224
BATCH_SIZE = 64
```

### 데이터 로드 {#loading-data}

여기서는 [tensorflow_datasets](https://www.tensorflow.org/datasets)(이하 TFDS)에서
데이터를 로드합니다.
Stanford Dogs 데이터 세트는
TFDS에서 [stanford_dogs](https://www.tensorflow.org/datasets/catalog/stanford_dogs)로 제공됩니다.
120개 견종에 속하는 20,580개의 이미지(트레이닝용 12,000개, 테스트용 8,580개)가 포함되어 있습니다.

아래 `dataset_name`을 변경하면,
[cifar10](https://www.tensorflow.org/datasets/catalog/cifar10),
[cifar100](https://www.tensorflow.org/datasets/catalog/cifar100),
[food101](https://www.tensorflow.org/datasets/catalog/food101) 등
TFDS의 다른 데이터 세트에 대해서도 이 노트북을 사용해 볼 수 있습니다.
이미지가 EfficientNet 입력 크기보다 훨씬 작은 경우, 입력 이미지를 업샘플링하면 됩니다.
[Tan and Le, 2019](https://arxiv.org/abs/1905.11946)에서
입력 이미지가 작아도 해상도가 높아지면 전이 학습 결과가 더 좋아진다는 것을 보여주었습니다.

```python
dataset_name = "stanford_dogs"
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train", "test"], with_info=True, as_supervised=True
)
NUM_CLASSES = ds_info.features["label"].num_classes
```

데이터 세트에 다양한 크기의 이미지가 포함되어 있는 경우,
우리는 공유되는 크기로 크기를 조정해야 합니다.
Stanford Dogs 데이터 세트에는 최소 200x200픽셀 크기의 이미지만 포함되어 있습니다.
여기서 우리는 이미지의 크기를 EfficientNet에 필요한 입력 크기로 조정합니다.

```python
size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
```

### 데이터 시각화 {#visualizing-the-data}

다음 코드는 처음 9개의 이미지와 해당 레이블을 보여줍니다.

```python
def format_label(label):
    string_label = label_info.int2str(label)
    return string_label.split("-")[1]


label_info = ds_info.features["label"]
for i, (image, label) in enumerate(ds_train.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("uint8"))
    plt.title("{}".format(format_label(label)))
    plt.axis("off")
```

![png](/images/examples/vision/image_classification_efficientnet_fine_tuning/image_classification_efficientnet_fine_tuning_9_0.png)

### 데이터 보강 {#data-augmentation}

이미지 보강을 위해 전처리 레이어 API를 사용할 수 있습니다.

```python
img_augmentation_layers = [
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.1),
]


def img_augmentation(images):
    for layer in img_augmentation_layers:
        images = layer(images)
    return images
```

이 `Sequential` 모델 객체는 나중에 빌드하는 모델의 일부로서 사용할 수도 있고,
모델에 데이터를 입력하기 전에 데이터를 사전 처리하는 함수로 사용할 수도 있습니다.
함수로 사용하면 보강된 이미지를 쉽게 시각화할 수 있습니다.
여기에서는 주어진 그림의 보강 결과의 9가지 예를 보여줍니다.

```python
for image, label in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(np.expand_dims(image.numpy(), axis=0))
        aug_img = np.array(aug_img)
        plt.imshow(aug_img[0].astype("uint8"))
        plt.title("{}".format(format_label(label)))
        plt.axis("off")
```

![png](/images/examples/vision/image_classification_efficientnet_fine_tuning/image_classification_efficientnet_fine_tuning_13_0.png)

### 입력 준비 {#prepare-inputs}

입력 데이터와 보강이 제대로 작동하는지 확인하면, 트레이닝을 위한 데이터 세트를 준비합니다.
입력 데이터의 크기를 균일한 `IMG_SIZE`로 조정합니다.
레이블은 one-hot(일명 카테고리형) 인코딩에 넣습니다. 데이터 세트가 배치 처리됩니다.

> 참고: `prefetch`와 `AUTOTUNE`은 경우에 따라 성능을 향상시킬 수 있지만,
> 환경과 사용되는 특정 데이터 세트에 따라 다릅니다.
> 데이터 파이프라인 성능에 대한 자세한 내용은
> 이 [가이드](https://www.tensorflow.org/guide/data_performance)를 참조하세요.

```python
# One-hot/카테고리형 인코딩 One-hot
def input_preprocess_train(image, label):
    image = img_augmentation(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def input_preprocess_test(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


ds_train = ds_train.map(input_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(input_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)
```

## 처음부터 모델 트레이닝하기 {#training-a-model-from-scratch}

처음부터 초기화된, 120개의 출력 클래스가 있는 EfficientNetB0을 빌드합니다:

> 참고: 정확도가 매우 느리게 증가하며, 과적합할 수 있습니다.

```python
model = EfficientNetB0(
    include_top=True,
    weights=None,
    classes=NUM_CLASSES,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

epochs = 40  # @param {type: "slider", min:10, max:100}
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "efficientnetb0"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃ Param # ┃ Connected to         ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (None, 224, 224,  │       0 │ -                    │
│ (InputLayer)        │ 3)                │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ rescaling           │ (None, 224, 224,  │       0 │ input_layer[0][0]    │
│ (Rescaling)         │ 3)                │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ normalization       │ (None, 224, 224,  │       7 │ rescaling[0][0]      │
│ (Normalization)     │ 3)                │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ stem_conv_pad       │ (None, 225, 225,  │       0 │ normalization[0][0]  │
│ (ZeroPadding2D)     │ 3)                │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ stem_conv (Conv2D)  │ (None, 112, 112,  │     864 │ stem_conv_pad[0][0]  │
│                     │ 32)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ stem_bn             │ (None, 112, 112,  │     128 │ stem_conv[0][0]      │
│ (BatchNormalizatio… │ 32)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ stem_activation     │ (None, 112, 112,  │       0 │ stem_bn[0][0]        │
│ (Activation)        │ 32)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block1a_dwconv      │ (None, 112, 112,  │     288 │ stem_activation[0][… │
│ (DepthwiseConv2D)   │ 32)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block1a_bn          │ (None, 112, 112,  │     128 │ block1a_dwconv[0][0] │
│ (BatchNormalizatio… │ 32)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block1a_activation  │ (None, 112, 112,  │       0 │ block1a_bn[0][0]     │
│ (Activation)        │ 32)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block1a_se_squeeze  │ (None, 32)        │       0 │ block1a_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block1a_se_reshape  │ (None, 1, 1, 32)  │       0 │ block1a_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block1a_se_reduce   │ (None, 1, 1, 8)   │     264 │ block1a_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block1a_se_expand   │ (None, 1, 1, 32)  │     288 │ block1a_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block1a_se_excite   │ (None, 112, 112,  │       0 │ block1a_activation[… │
│ (Multiply)          │ 32)               │         │ block1a_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block1a_project_co… │ (None, 112, 112,  │     512 │ block1a_se_excite[0… │
│ (Conv2D)            │ 16)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block1a_project_bn  │ (None, 112, 112,  │      64 │ block1a_project_con… │
│ (BatchNormalizatio… │ 16)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_expand_conv │ (None, 112, 112,  │   1,536 │ block1a_project_bn[… │
│ (Conv2D)            │ 96)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_expand_bn   │ (None, 112, 112,  │     384 │ block2a_expand_conv… │
│ (BatchNormalizatio… │ 96)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_expand_act… │ (None, 112, 112,  │       0 │ block2a_expand_bn[0… │
│ (Activation)        │ 96)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_dwconv_pad  │ (None, 113, 113,  │       0 │ block2a_expand_acti… │
│ (ZeroPadding2D)     │ 96)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_dwconv      │ (None, 56, 56,    │     864 │ block2a_dwconv_pad[… │
│ (DepthwiseConv2D)   │ 96)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_bn          │ (None, 56, 56,    │     384 │ block2a_dwconv[0][0] │
│ (BatchNormalizatio… │ 96)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_activation  │ (None, 56, 56,    │       0 │ block2a_bn[0][0]     │
│ (Activation)        │ 96)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_se_squeeze  │ (None, 96)        │       0 │ block2a_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_se_reshape  │ (None, 1, 1, 96)  │       0 │ block2a_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_se_reduce   │ (None, 1, 1, 4)   │     388 │ block2a_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_se_expand   │ (None, 1, 1, 96)  │     480 │ block2a_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_se_excite   │ (None, 56, 56,    │       0 │ block2a_activation[… │
│ (Multiply)          │ 96)               │         │ block2a_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_project_co… │ (None, 56, 56,    │   2,304 │ block2a_se_excite[0… │
│ (Conv2D)            │ 24)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2a_project_bn  │ (None, 56, 56,    │      96 │ block2a_project_con… │
│ (BatchNormalizatio… │ 24)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_expand_conv │ (None, 56, 56,    │   3,456 │ block2a_project_bn[… │
│ (Conv2D)            │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_expand_bn   │ (None, 56, 56,    │     576 │ block2b_expand_conv… │
│ (BatchNormalizatio… │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_expand_act… │ (None, 56, 56,    │       0 │ block2b_expand_bn[0… │
│ (Activation)        │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_dwconv      │ (None, 56, 56,    │   1,296 │ block2b_expand_acti… │
│ (DepthwiseConv2D)   │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_bn          │ (None, 56, 56,    │     576 │ block2b_dwconv[0][0] │
│ (BatchNormalizatio… │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_activation  │ (None, 56, 56,    │       0 │ block2b_bn[0][0]     │
│ (Activation)        │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_se_squeeze  │ (None, 144)       │       0 │ block2b_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_se_reshape  │ (None, 1, 1, 144) │       0 │ block2b_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_se_reduce   │ (None, 1, 1, 6)   │     870 │ block2b_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_se_expand   │ (None, 1, 1, 144) │   1,008 │ block2b_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_se_excite   │ (None, 56, 56,    │       0 │ block2b_activation[… │
│ (Multiply)          │ 144)              │         │ block2b_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_project_co… │ (None, 56, 56,    │   3,456 │ block2b_se_excite[0… │
│ (Conv2D)            │ 24)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_project_bn  │ (None, 56, 56,    │      96 │ block2b_project_con… │
│ (BatchNormalizatio… │ 24)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_drop        │ (None, 56, 56,    │       0 │ block2b_project_bn[… │
│ (Dropout)           │ 24)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block2b_add (Add)   │ (None, 56, 56,    │       0 │ block2b_drop[0][0],  │
│                     │ 24)               │         │ block2a_project_bn[… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_expand_conv │ (None, 56, 56,    │   3,456 │ block2b_add[0][0]    │
│ (Conv2D)            │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_expand_bn   │ (None, 56, 56,    │     576 │ block3a_expand_conv… │
│ (BatchNormalizatio… │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_expand_act… │ (None, 56, 56,    │       0 │ block3a_expand_bn[0… │
│ (Activation)        │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_dwconv_pad  │ (None, 59, 59,    │       0 │ block3a_expand_acti… │
│ (ZeroPadding2D)     │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_dwconv      │ (None, 28, 28,    │   3,600 │ block3a_dwconv_pad[… │
│ (DepthwiseConv2D)   │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_bn          │ (None, 28, 28,    │     576 │ block3a_dwconv[0][0] │
│ (BatchNormalizatio… │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_activation  │ (None, 28, 28,    │       0 │ block3a_bn[0][0]     │
│ (Activation)        │ 144)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_se_squeeze  │ (None, 144)       │       0 │ block3a_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_se_reshape  │ (None, 1, 1, 144) │       0 │ block3a_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_se_reduce   │ (None, 1, 1, 6)   │     870 │ block3a_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_se_expand   │ (None, 1, 1, 144) │   1,008 │ block3a_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_se_excite   │ (None, 28, 28,    │       0 │ block3a_activation[… │
│ (Multiply)          │ 144)              │         │ block3a_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_project_co… │ (None, 28, 28,    │   5,760 │ block3a_se_excite[0… │
│ (Conv2D)            │ 40)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3a_project_bn  │ (None, 28, 28,    │     160 │ block3a_project_con… │
│ (BatchNormalizatio… │ 40)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_expand_conv │ (None, 28, 28,    │   9,600 │ block3a_project_bn[… │
│ (Conv2D)            │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_expand_bn   │ (None, 28, 28,    │     960 │ block3b_expand_conv… │
│ (BatchNormalizatio… │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_expand_act… │ (None, 28, 28,    │       0 │ block3b_expand_bn[0… │
│ (Activation)        │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_dwconv      │ (None, 28, 28,    │   6,000 │ block3b_expand_acti… │
│ (DepthwiseConv2D)   │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_bn          │ (None, 28, 28,    │     960 │ block3b_dwconv[0][0] │
│ (BatchNormalizatio… │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_activation  │ (None, 28, 28,    │       0 │ block3b_bn[0][0]     │
│ (Activation)        │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_se_squeeze  │ (None, 240)       │       0 │ block3b_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_se_reshape  │ (None, 1, 1, 240) │       0 │ block3b_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_se_reduce   │ (None, 1, 1, 10)  │   2,410 │ block3b_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_se_expand   │ (None, 1, 1, 240) │   2,640 │ block3b_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_se_excite   │ (None, 28, 28,    │       0 │ block3b_activation[… │
│ (Multiply)          │ 240)              │         │ block3b_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_project_co… │ (None, 28, 28,    │   9,600 │ block3b_se_excite[0… │
│ (Conv2D)            │ 40)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_project_bn  │ (None, 28, 28,    │     160 │ block3b_project_con… │
│ (BatchNormalizatio… │ 40)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_drop        │ (None, 28, 28,    │       0 │ block3b_project_bn[… │
│ (Dropout)           │ 40)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block3b_add (Add)   │ (None, 28, 28,    │       0 │ block3b_drop[0][0],  │
│                     │ 40)               │         │ block3a_project_bn[… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_expand_conv │ (None, 28, 28,    │   9,600 │ block3b_add[0][0]    │
│ (Conv2D)            │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_expand_bn   │ (None, 28, 28,    │     960 │ block4a_expand_conv… │
│ (BatchNormalizatio… │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_expand_act… │ (None, 28, 28,    │       0 │ block4a_expand_bn[0… │
│ (Activation)        │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_dwconv_pad  │ (None, 29, 29,    │       0 │ block4a_expand_acti… │
│ (ZeroPadding2D)     │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_dwconv      │ (None, 14, 14,    │   2,160 │ block4a_dwconv_pad[… │
│ (DepthwiseConv2D)   │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_bn          │ (None, 14, 14,    │     960 │ block4a_dwconv[0][0] │
│ (BatchNormalizatio… │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_activation  │ (None, 14, 14,    │       0 │ block4a_bn[0][0]     │
│ (Activation)        │ 240)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_se_squeeze  │ (None, 240)       │       0 │ block4a_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_se_reshape  │ (None, 1, 1, 240) │       0 │ block4a_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_se_reduce   │ (None, 1, 1, 10)  │   2,410 │ block4a_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_se_expand   │ (None, 1, 1, 240) │   2,640 │ block4a_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_se_excite   │ (None, 14, 14,    │       0 │ block4a_activation[… │
│ (Multiply)          │ 240)              │         │ block4a_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_project_co… │ (None, 14, 14,    │  19,200 │ block4a_se_excite[0… │
│ (Conv2D)            │ 80)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4a_project_bn  │ (None, 14, 14,    │     320 │ block4a_project_con… │
│ (BatchNormalizatio… │ 80)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_expand_conv │ (None, 14, 14,    │  38,400 │ block4a_project_bn[… │
│ (Conv2D)            │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_expand_bn   │ (None, 14, 14,    │   1,920 │ block4b_expand_conv… │
│ (BatchNormalizatio… │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_expand_act… │ (None, 14, 14,    │       0 │ block4b_expand_bn[0… │
│ (Activation)        │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_dwconv      │ (None, 14, 14,    │   4,320 │ block4b_expand_acti… │
│ (DepthwiseConv2D)   │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_bn          │ (None, 14, 14,    │   1,920 │ block4b_dwconv[0][0] │
│ (BatchNormalizatio… │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_activation  │ (None, 14, 14,    │       0 │ block4b_bn[0][0]     │
│ (Activation)        │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_se_squeeze  │ (None, 480)       │       0 │ block4b_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_se_reshape  │ (None, 1, 1, 480) │       0 │ block4b_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_se_reduce   │ (None, 1, 1, 20)  │   9,620 │ block4b_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_se_expand   │ (None, 1, 1, 480) │  10,080 │ block4b_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_se_excite   │ (None, 14, 14,    │       0 │ block4b_activation[… │
│ (Multiply)          │ 480)              │         │ block4b_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_project_co… │ (None, 14, 14,    │  38,400 │ block4b_se_excite[0… │
│ (Conv2D)            │ 80)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_project_bn  │ (None, 14, 14,    │     320 │ block4b_project_con… │
│ (BatchNormalizatio… │ 80)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_drop        │ (None, 14, 14,    │       0 │ block4b_project_bn[… │
│ (Dropout)           │ 80)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4b_add (Add)   │ (None, 14, 14,    │       0 │ block4b_drop[0][0],  │
│                     │ 80)               │         │ block4a_project_bn[… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_expand_conv │ (None, 14, 14,    │  38,400 │ block4b_add[0][0]    │
│ (Conv2D)            │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_expand_bn   │ (None, 14, 14,    │   1,920 │ block4c_expand_conv… │
│ (BatchNormalizatio… │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_expand_act… │ (None, 14, 14,    │       0 │ block4c_expand_bn[0… │
│ (Activation)        │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_dwconv      │ (None, 14, 14,    │   4,320 │ block4c_expand_acti… │
│ (DepthwiseConv2D)   │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_bn          │ (None, 14, 14,    │   1,920 │ block4c_dwconv[0][0] │
│ (BatchNormalizatio… │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_activation  │ (None, 14, 14,    │       0 │ block4c_bn[0][0]     │
│ (Activation)        │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_se_squeeze  │ (None, 480)       │       0 │ block4c_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_se_reshape  │ (None, 1, 1, 480) │       0 │ block4c_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_se_reduce   │ (None, 1, 1, 20)  │   9,620 │ block4c_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_se_expand   │ (None, 1, 1, 480) │  10,080 │ block4c_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_se_excite   │ (None, 14, 14,    │       0 │ block4c_activation[… │
│ (Multiply)          │ 480)              │         │ block4c_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_project_co… │ (None, 14, 14,    │  38,400 │ block4c_se_excite[0… │
│ (Conv2D)            │ 80)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_project_bn  │ (None, 14, 14,    │     320 │ block4c_project_con… │
│ (BatchNormalizatio… │ 80)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_drop        │ (None, 14, 14,    │       0 │ block4c_project_bn[… │
│ (Dropout)           │ 80)               │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block4c_add (Add)   │ (None, 14, 14,    │       0 │ block4c_drop[0][0],  │
│                     │ 80)               │         │ block4b_add[0][0]    │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_expand_conv │ (None, 14, 14,    │  38,400 │ block4c_add[0][0]    │
│ (Conv2D)            │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_expand_bn   │ (None, 14, 14,    │   1,920 │ block5a_expand_conv… │
│ (BatchNormalizatio… │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_expand_act… │ (None, 14, 14,    │       0 │ block5a_expand_bn[0… │
│ (Activation)        │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_dwconv      │ (None, 14, 14,    │  12,000 │ block5a_expand_acti… │
│ (DepthwiseConv2D)   │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_bn          │ (None, 14, 14,    │   1,920 │ block5a_dwconv[0][0] │
│ (BatchNormalizatio… │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_activation  │ (None, 14, 14,    │       0 │ block5a_bn[0][0]     │
│ (Activation)        │ 480)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_se_squeeze  │ (None, 480)       │       0 │ block5a_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_se_reshape  │ (None, 1, 1, 480) │       0 │ block5a_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_se_reduce   │ (None, 1, 1, 20)  │   9,620 │ block5a_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_se_expand   │ (None, 1, 1, 480) │  10,080 │ block5a_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_se_excite   │ (None, 14, 14,    │       0 │ block5a_activation[… │
│ (Multiply)          │ 480)              │         │ block5a_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_project_co… │ (None, 14, 14,    │  53,760 │ block5a_se_excite[0… │
│ (Conv2D)            │ 112)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5a_project_bn  │ (None, 14, 14,    │     448 │ block5a_project_con… │
│ (BatchNormalizatio… │ 112)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_expand_conv │ (None, 14, 14,    │  75,264 │ block5a_project_bn[… │
│ (Conv2D)            │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_expand_bn   │ (None, 14, 14,    │   2,688 │ block5b_expand_conv… │
│ (BatchNormalizatio… │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_expand_act… │ (None, 14, 14,    │       0 │ block5b_expand_bn[0… │
│ (Activation)        │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_dwconv      │ (None, 14, 14,    │  16,800 │ block5b_expand_acti… │
│ (DepthwiseConv2D)   │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_bn          │ (None, 14, 14,    │   2,688 │ block5b_dwconv[0][0] │
│ (BatchNormalizatio… │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_activation  │ (None, 14, 14,    │       0 │ block5b_bn[0][0]     │
│ (Activation)        │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_se_squeeze  │ (None, 672)       │       0 │ block5b_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_se_reshape  │ (None, 1, 1, 672) │       0 │ block5b_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_se_reduce   │ (None, 1, 1, 28)  │  18,844 │ block5b_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_se_expand   │ (None, 1, 1, 672) │  19,488 │ block5b_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_se_excite   │ (None, 14, 14,    │       0 │ block5b_activation[… │
│ (Multiply)          │ 672)              │         │ block5b_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_project_co… │ (None, 14, 14,    │  75,264 │ block5b_se_excite[0… │
│ (Conv2D)            │ 112)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_project_bn  │ (None, 14, 14,    │     448 │ block5b_project_con… │
│ (BatchNormalizatio… │ 112)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_drop        │ (None, 14, 14,    │       0 │ block5b_project_bn[… │
│ (Dropout)           │ 112)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5b_add (Add)   │ (None, 14, 14,    │       0 │ block5b_drop[0][0],  │
│                     │ 112)              │         │ block5a_project_bn[… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_expand_conv │ (None, 14, 14,    │  75,264 │ block5b_add[0][0]    │
│ (Conv2D)            │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_expand_bn   │ (None, 14, 14,    │   2,688 │ block5c_expand_conv… │
│ (BatchNormalizatio… │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_expand_act… │ (None, 14, 14,    │       0 │ block5c_expand_bn[0… │
│ (Activation)        │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_dwconv      │ (None, 14, 14,    │  16,800 │ block5c_expand_acti… │
│ (DepthwiseConv2D)   │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_bn          │ (None, 14, 14,    │   2,688 │ block5c_dwconv[0][0] │
│ (BatchNormalizatio… │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_activation  │ (None, 14, 14,    │       0 │ block5c_bn[0][0]     │
│ (Activation)        │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_se_squeeze  │ (None, 672)       │       0 │ block5c_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_se_reshape  │ (None, 1, 1, 672) │       0 │ block5c_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_se_reduce   │ (None, 1, 1, 28)  │  18,844 │ block5c_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_se_expand   │ (None, 1, 1, 672) │  19,488 │ block5c_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_se_excite   │ (None, 14, 14,    │       0 │ block5c_activation[… │
│ (Multiply)          │ 672)              │         │ block5c_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_project_co… │ (None, 14, 14,    │  75,264 │ block5c_se_excite[0… │
│ (Conv2D)            │ 112)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_project_bn  │ (None, 14, 14,    │     448 │ block5c_project_con… │
│ (BatchNormalizatio… │ 112)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_drop        │ (None, 14, 14,    │       0 │ block5c_project_bn[… │
│ (Dropout)           │ 112)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block5c_add (Add)   │ (None, 14, 14,    │       0 │ block5c_drop[0][0],  │
│                     │ 112)              │         │ block5b_add[0][0]    │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_expand_conv │ (None, 14, 14,    │  75,264 │ block5c_add[0][0]    │
│ (Conv2D)            │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_expand_bn   │ (None, 14, 14,    │   2,688 │ block6a_expand_conv… │
│ (BatchNormalizatio… │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_expand_act… │ (None, 14, 14,    │       0 │ block6a_expand_bn[0… │
│ (Activation)        │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_dwconv_pad  │ (None, 17, 17,    │       0 │ block6a_expand_acti… │
│ (ZeroPadding2D)     │ 672)              │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_dwconv      │ (None, 7, 7, 672) │  16,800 │ block6a_dwconv_pad[… │
│ (DepthwiseConv2D)   │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_bn          │ (None, 7, 7, 672) │   2,688 │ block6a_dwconv[0][0] │
│ (BatchNormalizatio… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_activation  │ (None, 7, 7, 672) │       0 │ block6a_bn[0][0]     │
│ (Activation)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_se_squeeze  │ (None, 672)       │       0 │ block6a_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_se_reshape  │ (None, 1, 1, 672) │       0 │ block6a_se_squeeze[… │
│ (Reshape)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_se_reduce   │ (None, 1, 1, 28)  │  18,844 │ block6a_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_se_expand   │ (None, 1, 1, 672) │  19,488 │ block6a_se_reduce[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_se_excite   │ (None, 7, 7, 672) │       0 │ block6a_activation[… │
│ (Multiply)          │                   │         │ block6a_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_project_co… │ (None, 7, 7, 192) │ 129,024 │ block6a_se_excite[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6a_project_bn  │ (None, 7, 7, 192) │     768 │ block6a_project_con… │
│ (BatchNormalizatio… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_expand_conv │ (None, 7, 7,      │ 221,184 │ block6a_project_bn[… │
│ (Conv2D)            │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_expand_bn   │ (None, 7, 7,      │   4,608 │ block6b_expand_conv… │
│ (BatchNormalizatio… │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_expand_act… │ (None, 7, 7,      │       0 │ block6b_expand_bn[0… │
│ (Activation)        │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_dwconv      │ (None, 7, 7,      │  28,800 │ block6b_expand_acti… │
│ (DepthwiseConv2D)   │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_bn          │ (None, 7, 7,      │   4,608 │ block6b_dwconv[0][0] │
│ (BatchNormalizatio… │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_activation  │ (None, 7, 7,      │       0 │ block6b_bn[0][0]     │
│ (Activation)        │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_se_squeeze  │ (None, 1152)      │       0 │ block6b_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_se_reshape  │ (None, 1, 1,      │       0 │ block6b_se_squeeze[… │
│ (Reshape)           │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_se_reduce   │ (None, 1, 1, 48)  │  55,344 │ block6b_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_se_expand   │ (None, 1, 1,      │  56,448 │ block6b_se_reduce[0… │
│ (Conv2D)            │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_se_excite   │ (None, 7, 7,      │       0 │ block6b_activation[… │
│ (Multiply)          │ 1152)             │         │ block6b_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_project_co… │ (None, 7, 7, 192) │ 221,184 │ block6b_se_excite[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_project_bn  │ (None, 7, 7, 192) │     768 │ block6b_project_con… │
│ (BatchNormalizatio… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_drop        │ (None, 7, 7, 192) │       0 │ block6b_project_bn[… │
│ (Dropout)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6b_add (Add)   │ (None, 7, 7, 192) │       0 │ block6b_drop[0][0],  │
│                     │                   │         │ block6a_project_bn[… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_expand_conv │ (None, 7, 7,      │ 221,184 │ block6b_add[0][0]    │
│ (Conv2D)            │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_expand_bn   │ (None, 7, 7,      │   4,608 │ block6c_expand_conv… │
│ (BatchNormalizatio… │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_expand_act… │ (None, 7, 7,      │       0 │ block6c_expand_bn[0… │
│ (Activation)        │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_dwconv      │ (None, 7, 7,      │  28,800 │ block6c_expand_acti… │
│ (DepthwiseConv2D)   │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_bn          │ (None, 7, 7,      │   4,608 │ block6c_dwconv[0][0] │
│ (BatchNormalizatio… │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_activation  │ (None, 7, 7,      │       0 │ block6c_bn[0][0]     │
│ (Activation)        │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_se_squeeze  │ (None, 1152)      │       0 │ block6c_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_se_reshape  │ (None, 1, 1,      │       0 │ block6c_se_squeeze[… │
│ (Reshape)           │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_se_reduce   │ (None, 1, 1, 48)  │  55,344 │ block6c_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_se_expand   │ (None, 1, 1,      │  56,448 │ block6c_se_reduce[0… │
│ (Conv2D)            │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_se_excite   │ (None, 7, 7,      │       0 │ block6c_activation[… │
│ (Multiply)          │ 1152)             │         │ block6c_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_project_co… │ (None, 7, 7, 192) │ 221,184 │ block6c_se_excite[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_project_bn  │ (None, 7, 7, 192) │     768 │ block6c_project_con… │
│ (BatchNormalizatio… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_drop        │ (None, 7, 7, 192) │       0 │ block6c_project_bn[… │
│ (Dropout)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6c_add (Add)   │ (None, 7, 7, 192) │       0 │ block6c_drop[0][0],  │
│                     │                   │         │ block6b_add[0][0]    │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_expand_conv │ (None, 7, 7,      │ 221,184 │ block6c_add[0][0]    │
│ (Conv2D)            │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_expand_bn   │ (None, 7, 7,      │   4,608 │ block6d_expand_conv… │
│ (BatchNormalizatio… │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_expand_act… │ (None, 7, 7,      │       0 │ block6d_expand_bn[0… │
│ (Activation)        │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_dwconv      │ (None, 7, 7,      │  28,800 │ block6d_expand_acti… │
│ (DepthwiseConv2D)   │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_bn          │ (None, 7, 7,      │   4,608 │ block6d_dwconv[0][0] │
│ (BatchNormalizatio… │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_activation  │ (None, 7, 7,      │       0 │ block6d_bn[0][0]     │
│ (Activation)        │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_se_squeeze  │ (None, 1152)      │       0 │ block6d_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_se_reshape  │ (None, 1, 1,      │       0 │ block6d_se_squeeze[… │
│ (Reshape)           │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_se_reduce   │ (None, 1, 1, 48)  │  55,344 │ block6d_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_se_expand   │ (None, 1, 1,      │  56,448 │ block6d_se_reduce[0… │
│ (Conv2D)            │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_se_excite   │ (None, 7, 7,      │       0 │ block6d_activation[… │
│ (Multiply)          │ 1152)             │         │ block6d_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_project_co… │ (None, 7, 7, 192) │ 221,184 │ block6d_se_excite[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_project_bn  │ (None, 7, 7, 192) │     768 │ block6d_project_con… │
│ (BatchNormalizatio… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_drop        │ (None, 7, 7, 192) │       0 │ block6d_project_bn[… │
│ (Dropout)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block6d_add (Add)   │ (None, 7, 7, 192) │       0 │ block6d_drop[0][0],  │
│                     │                   │         │ block6c_add[0][0]    │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_expand_conv │ (None, 7, 7,      │ 221,184 │ block6d_add[0][0]    │
│ (Conv2D)            │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_expand_bn   │ (None, 7, 7,      │   4,608 │ block7a_expand_conv… │
│ (BatchNormalizatio… │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_expand_act… │ (None, 7, 7,      │       0 │ block7a_expand_bn[0… │
│ (Activation)        │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_dwconv      │ (None, 7, 7,      │  10,368 │ block7a_expand_acti… │
│ (DepthwiseConv2D)   │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_bn          │ (None, 7, 7,      │   4,608 │ block7a_dwconv[0][0] │
│ (BatchNormalizatio… │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_activation  │ (None, 7, 7,      │       0 │ block7a_bn[0][0]     │
│ (Activation)        │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_se_squeeze  │ (None, 1152)      │       0 │ block7a_activation[… │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_se_reshape  │ (None, 1, 1,      │       0 │ block7a_se_squeeze[… │
│ (Reshape)           │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_se_reduce   │ (None, 1, 1, 48)  │  55,344 │ block7a_se_reshape[… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_se_expand   │ (None, 1, 1,      │  56,448 │ block7a_se_reduce[0… │
│ (Conv2D)            │ 1152)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_se_excite   │ (None, 7, 7,      │       0 │ block7a_activation[… │
│ (Multiply)          │ 1152)             │         │ block7a_se_expand[0… │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_project_co… │ (None, 7, 7, 320) │ 368,640 │ block7a_se_excite[0… │
│ (Conv2D)            │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ block7a_project_bn  │ (None, 7, 7, 320) │   1,280 │ block7a_project_con… │
│ (BatchNormalizatio… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ top_conv (Conv2D)   │ (None, 7, 7,      │ 409,600 │ block7a_project_bn[… │
│                     │ 1280)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ top_bn              │ (None, 7, 7,      │   5,120 │ top_conv[0][0]       │
│ (BatchNormalizatio… │ 1280)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ top_activation      │ (None, 7, 7,      │       0 │ top_bn[0][0]         │
│ (Activation)        │ 1280)             │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ avg_pool            │ (None, 1280)      │       0 │ top_activation[0][0] │
│ (GlobalAveragePool… │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ top_dropout         │ (None, 1280)      │       0 │ avg_pool[0][0]       │
│ (Dropout)           │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ predictions (Dense) │ (None, 120)       │ 153,720 │ top_dropout[0][0]    │
└─────────────────────┴───────────────────┴─────────┴──────────────────────┘
 Total params: 4,203,291 (16.03 MB)
 Trainable params: 4,161,268 (15.87 MB)
 Non-trainable params: 42,023 (164.16 KB)
```

```plain
Epoch 1/40
   1/187 [37m━━━━━━━━━━━━━━━━━━━━  5:30:13 107s/step - accuracy: 0.0000e+00 - loss: 5.1065

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1700241724.682725 1549299 device_compiler.h:187] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

 187/187 ━━━━━━━━━━━━━━━━━━━━ 200s 501ms/step - accuracy: 0.0097 - loss: 5.0567 - val_accuracy: 0.0100 - val_loss: 4.9278
Epoch 2/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 95s 507ms/step - accuracy: 0.0214 - loss: 4.6918 - val_accuracy: 0.0141 - val_loss: 5.5380
Epoch 3/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 89s 474ms/step - accuracy: 0.0298 - loss: 4.4749 - val_accuracy: 0.0375 - val_loss: 4.4576
Epoch 4/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 90s 479ms/step - accuracy: 0.0423 - loss: 4.3206 - val_accuracy: 0.0391 - val_loss: 4.9898
Epoch 5/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 89s 473ms/step - accuracy: 0.0458 - loss: 4.2312 - val_accuracy: 0.0416 - val_loss: 4.3210
Epoch 6/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 141s 470ms/step - accuracy: 0.0579 - loss: 4.1162 - val_accuracy: 0.0540 - val_loss: 4.3371
Epoch 7/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 89s 476ms/step - accuracy: 0.0679 - loss: 4.0150 - val_accuracy: 0.0786 - val_loss: 3.9759
Epoch 8/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 89s 477ms/step - accuracy: 0.0828 - loss: 3.9147 - val_accuracy: 0.0651 - val_loss: 4.1641
Epoch 9/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 142s 475ms/step - accuracy: 0.0932 - loss: 3.8297 - val_accuracy: 0.0928 - val_loss: 3.8985
Epoch 10/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 88s 472ms/step - accuracy: 0.1092 - loss: 3.7321 - val_accuracy: 0.0946 - val_loss: 3.8618
Epoch 11/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 89s 476ms/step - accuracy: 0.1245 - loss: 3.6451 - val_accuracy: 0.0880 - val_loss: 3.9584
Epoch 12/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 92s 493ms/step - accuracy: 0.1457 - loss: 3.5514 - val_accuracy: 0.1096 - val_loss: 3.8184
Epoch 13/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 88s 471ms/step - accuracy: 0.1606 - loss: 3.4654 - val_accuracy: 0.1118 - val_loss: 3.8059
Epoch 14/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 87s 464ms/step - accuracy: 0.1660 - loss: 3.3826 - val_accuracy: 0.1472 - val_loss: 3.5726
Epoch 15/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 146s 485ms/step - accuracy: 0.1815 - loss: 3.2935 - val_accuracy: 0.1154 - val_loss: 3.8134
Epoch 16/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 87s 466ms/step - accuracy: 0.1942 - loss: 3.2218 - val_accuracy: 0.1540 - val_loss: 3.5051
Epoch 17/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 88s 471ms/step - accuracy: 0.2131 - loss: 3.1427 - val_accuracy: 0.1381 - val_loss: 3.7206
Epoch 18/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 87s 467ms/step - accuracy: 0.2264 - loss: 3.0461 - val_accuracy: 0.1707 - val_loss: 3.4122
Epoch 19/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 88s 470ms/step - accuracy: 0.2401 - loss: 2.9821 - val_accuracy: 0.1515 - val_loss: 3.6481
Epoch 20/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 88s 469ms/step - accuracy: 0.2613 - loss: 2.8815 - val_accuracy: 0.1783 - val_loss: 3.4767
Epoch 21/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 91s 485ms/step - accuracy: 0.2741 - loss: 2.8102 - val_accuracy: 0.1927 - val_loss: 3.3183
Epoch 22/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 90s 477ms/step - accuracy: 0.2892 - loss: 2.7408 - val_accuracy: 0.1859 - val_loss: 3.4887
Epoch 23/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 91s 485ms/step - accuracy: 0.3093 - loss: 2.6526 - val_accuracy: 0.1924 - val_loss: 3.4622
Epoch 24/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 92s 491ms/step - accuracy: 0.3201 - loss: 2.5750 - val_accuracy: 0.2253 - val_loss: 3.1873
Epoch 25/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 95s 508ms/step - accuracy: 0.3280 - loss: 2.5150 - val_accuracy: 0.2148 - val_loss: 3.3391
Epoch 26/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 92s 490ms/step - accuracy: 0.3465 - loss: 2.4402 - val_accuracy: 0.2270 - val_loss: 3.2679
Epoch 27/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 93s 494ms/step - accuracy: 0.3735 - loss: 2.3199 - val_accuracy: 0.2080 - val_loss: 3.5687
Epoch 28/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 89s 476ms/step - accuracy: 0.3837 - loss: 2.2645 - val_accuracy: 0.2374 - val_loss: 3.3592
Epoch 29/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 142s 474ms/step - accuracy: 0.3962 - loss: 2.2110 - val_accuracy: 0.2008 - val_loss: 3.6071
Epoch 30/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 87s 466ms/step - accuracy: 0.4175 - loss: 2.1086 - val_accuracy: 0.2302 - val_loss: 3.4161
Epoch 31/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 87s 465ms/step - accuracy: 0.4359 - loss: 2.0610 - val_accuracy: 0.2231 - val_loss: 3.5957
Epoch 32/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 148s 498ms/step - accuracy: 0.4463 - loss: 1.9866 - val_accuracy: 0.2234 - val_loss: 3.7263
Epoch 33/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 92s 489ms/step - accuracy: 0.4613 - loss: 1.8821 - val_accuracy: 0.2239 - val_loss: 3.6929
Epoch 34/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 139s 475ms/step - accuracy: 0.4925 - loss: 1.7858 - val_accuracy: 0.2238 - val_loss: 3.8351
Epoch 35/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 91s 485ms/step - accuracy: 0.5105 - loss: 1.7074 - val_accuracy: 0.1930 - val_loss: 4.1941
Epoch 36/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 140s 474ms/step - accuracy: 0.5334 - loss: 1.6256 - val_accuracy: 0.2098 - val_loss: 4.1464
Epoch 37/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 87s 464ms/step - accuracy: 0.5504 - loss: 1.5603 - val_accuracy: 0.2306 - val_loss: 4.0215
Epoch 38/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 90s 480ms/step - accuracy: 0.5736 - loss: 1.4419 - val_accuracy: 0.2240 - val_loss: 4.1604
Epoch 39/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 91s 486ms/step - accuracy: 0.6025 - loss: 1.3612 - val_accuracy: 0.2344 - val_loss: 4.0505
Epoch 40/40
 187/187 ━━━━━━━━━━━━━━━━━━━━ 89s 474ms/step - accuracy: 0.6199 - loss: 1.2889 - val_accuracy: 0.2151 - val_loss: 4.3660
```

{{% /details %}}

모델 트레이닝은 비교적 빠릅니다.
따라서, 처음부터 원하는 데이터 세트에 대해
EfficientNet을 트레이닝하는 것이 쉬워 보일 수 있습니다.
그러나, 소규모 데이터 세트, 특히 CIFAR-100과 같이 해상도가 낮은 데이터 세트에 대해,
EfficientNet을 트레이닝할 경우, 과적합이라는 중대한 문제에 직면하게 됩니다.

따라서 처음부터 트레이닝하려면 하이퍼파라미터를 매우 신중하게 선택해야 하며,
적절한 정규화를 찾기가 어렵습니다.
이것은 또한 리소스도 훨씬 더 많이 요구됩니다.
트레이닝과 검증 정확도를 그래프로 그려보면,
검증 정확도가 낮은 값에서 정체되는 것을 알 수 있습니다.

```python
import matplotlib.pyplot as plt


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


plot_hist(hist)
```

![png](/images/examples/vision/image_classification_efficientnet_fine_tuning/image_classification_efficientnet_fine_tuning_19_0.png)

## 사전 트레이닝된 가중치를 통한 전이 학습 {#transfer-learning-from-pre-trained-weights}

여기서는 미리 트레이닝된 ImageNet 가중치로 모델을 초기화하고, 자체 데이터 세트에서 미세 조정합니다.

```python
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # 사전 트레이닝된 가중치 동결
    model.trainable = False

    # 톱만 다시 빌드
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # 컴파일
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
```

전이 학습의 첫 번째 단계는 모든 레이어를 동결하고 최상위 레이어만 트레이닝하는 것입니다.
이 단계에서는, 비교적 큰 학습률(1e-2)을 사용할 수 있습니다.
일반적으로 검증 정확도와 손실이 트레이닝 정확도와 손실보다 낫다는 점에 유의하세요.
이는 정규화가 강력하여, 트레이닝 시의 메트릭만 억제하기 때문입니다.

학습률 선택에 따라 수렴에 최대 50개의 에포크가 소요될 수 있습니다.
이미지 보강 레이어를 적용하지 않은 경우, 검증 정확도는 최대 60%에 불과할 수 있습니다.

```python
model = build_model(num_classes=NUM_CLASSES)

epochs = 25  # @param {type: "slider", min:8, max:80}
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test)
plot_hist(hist)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 108s 432ms/step - accuracy: 0.2654 - loss: 4.3710 - val_accuracy: 0.6888 - val_loss: 1.0875
Epoch 2/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 119s 412ms/step - accuracy: 0.4863 - loss: 2.0996 - val_accuracy: 0.7282 - val_loss: 0.9072
Epoch 3/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 78s 416ms/step - accuracy: 0.5422 - loss: 1.7120 - val_accuracy: 0.7411 - val_loss: 0.8574
Epoch 4/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 77s 412ms/step - accuracy: 0.5509 - loss: 1.6472 - val_accuracy: 0.7451 - val_loss: 0.8457
Epoch 5/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 81s 431ms/step - accuracy: 0.5744 - loss: 1.5373 - val_accuracy: 0.7424 - val_loss: 0.8649
Epoch 6/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 78s 417ms/step - accuracy: 0.5715 - loss: 1.5595 - val_accuracy: 0.7374 - val_loss: 0.8736
Epoch 7/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 81s 432ms/step - accuracy: 0.5802 - loss: 1.5045 - val_accuracy: 0.7430 - val_loss: 0.8675
Epoch 8/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 77s 411ms/step - accuracy: 0.5839 - loss: 1.4972 - val_accuracy: 0.7392 - val_loss: 0.8647
Epoch 9/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 77s 411ms/step - accuracy: 0.5929 - loss: 1.4699 - val_accuracy: 0.7508 - val_loss: 0.8634
Epoch 10/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 82s 437ms/step - accuracy: 0.6040 - loss: 1.4442 - val_accuracy: 0.7520 - val_loss: 0.8480
Epoch 11/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 78s 416ms/step - accuracy: 0.5972 - loss: 1.4626 - val_accuracy: 0.7379 - val_loss: 0.8879
Epoch 12/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 79s 421ms/step - accuracy: 0.5965 - loss: 1.4700 - val_accuracy: 0.7383 - val_loss: 0.9409
Epoch 13/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 82s 420ms/step - accuracy: 0.6034 - loss: 1.4533 - val_accuracy: 0.7474 - val_loss: 0.8922
Epoch 14/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 81s 435ms/step - accuracy: 0.6053 - loss: 1.4170 - val_accuracy: 0.7416 - val_loss: 0.9119
Epoch 15/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 77s 411ms/step - accuracy: 0.6059 - loss: 1.4125 - val_accuracy: 0.7406 - val_loss: 0.9205
Epoch 16/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 82s 438ms/step - accuracy: 0.5979 - loss: 1.4554 - val_accuracy: 0.7392 - val_loss: 0.9120
Epoch 17/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 77s 411ms/step - accuracy: 0.6081 - loss: 1.4089 - val_accuracy: 0.7423 - val_loss: 0.9305
Epoch 18/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 82s 436ms/step - accuracy: 0.6041 - loss: 1.4390 - val_accuracy: 0.7380 - val_loss: 0.9644
Epoch 19/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 79s 417ms/step - accuracy: 0.6018 - loss: 1.4324 - val_accuracy: 0.7439 - val_loss: 0.9129
Epoch 20/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 81s 430ms/step - accuracy: 0.6057 - loss: 1.4342 - val_accuracy: 0.7305 - val_loss: 0.9463
Epoch 21/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 77s 410ms/step - accuracy: 0.6209 - loss: 1.3824 - val_accuracy: 0.7410 - val_loss: 0.9503
Epoch 22/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 78s 419ms/step - accuracy: 0.6170 - loss: 1.4246 - val_accuracy: 0.7336 - val_loss: 0.9606
Epoch 23/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 85s 455ms/step - accuracy: 0.6153 - loss: 1.4009 - val_accuracy: 0.7334 - val_loss: 0.9520
Epoch 24/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 82s 438ms/step - accuracy: 0.6051 - loss: 1.4343 - val_accuracy: 0.7435 - val_loss: 0.9403
Epoch 25/25
 187/187 ━━━━━━━━━━━━━━━━━━━━ 138s 416ms/step - accuracy: 0.6065 - loss: 1.4131 - val_accuracy: 0.7456 - val_loss: 0.9307
```

{{% /details %}}

![png](/images/examples/vision/image_classification_efficientnet_fine_tuning/image_classification_efficientnet_fine_tuning_23_1.png)

두 번째 단계는 여러 레이어의 동결을 해제하고 더 작은 학습률을 사용하여 모델을 맞추는 것입니다.
이 예에서는 모든 레이어를 동결 해제하는 것을 보여드리지만,
특정 데이터 세트에 따라 전체 레이어 중 일부만 동결 해제하는 것이 바람직할 수도 있습니다.

사전 트레이닝된 모델을 사용한 특성 추출이 충분히 잘 작동하는 경우,
이 단계는 검증 정확도를 매우 제한적으로 향상시킬 수 있습니다.
우리의 경우에는 이미 ImageNet 사전 트레이닝된 모델이 많은 양의 개에 노출되어 있었기 때문에
약간의 개선만 있었습니다.

반면, ImageNet으로부터의 다른 데이터 세트에 대해 사전 트레이닝된 가중치를 사용하는 경우,
특성 추출기 역시 상당 부분 조정해야 하므로, 이 미세 조정 단계가 매우 중요할 수 있습니다.
이러한 상황은 CIFAR-100 데이터 세트를 선택하면 확인할 수 있는데,
이 경우 미세 조정을 통해 검증 정확도가 약 10% 향상되어,
`EfficientNetB0`에서 80%를 통과할 수 있습니다.

모델 동결/해제에 대한 참고 사항: `Model`의 `trainable`을 설정하면,
`Model`에 속한 모든 레이어가 동시에 동일한 `trainable` 속성으로 설정됩니다.
각 레이어는 레이어 자체와 레이어가 포함된 모델이 모두 트레이닝 가능한 경우에만 트레이닝 가능할 수 있습니다.
따라서, 모델을 부분적으로 동결/동결 해제해야 하는 경우,
모델의 `trainable` 속성이 `True`로 설정되어 있는지 확인해야 합니다.

```python
def unfreeze_model(model):
    # 상위 20개 레이어는 동결 해제하고 BatchNorm 레이어는 고정된 상태로 둡니다.
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


unfreeze_model(model)

epochs = 4  # @param {type: "slider", min:4, max:10}
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test)
plot_hist(hist)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/4
 187/187 ━━━━━━━━━━━━━━━━━━━━ 111s 442ms/step - accuracy: 0.6310 - loss: 1.3425 - val_accuracy: 0.7565 - val_loss: 0.8874
Epoch 2/4
 187/187 ━━━━━━━━━━━━━━━━━━━━ 77s 413ms/step - accuracy: 0.6518 - loss: 1.2755 - val_accuracy: 0.7635 - val_loss: 0.8588
Epoch 3/4
 187/187 ━━━━━━━━━━━━━━━━━━━━ 82s 437ms/step - accuracy: 0.6491 - loss: 1.2426 - val_accuracy: 0.7663 - val_loss: 0.8419
Epoch 4/4
 187/187 ━━━━━━━━━━━━━━━━━━━━ 79s 419ms/step - accuracy: 0.6625 - loss: 1.1775 - val_accuracy: 0.7701 - val_loss: 0.8284
```

{{% /details %}}

![png](/images/examples/vision/image_classification_efficientnet_fine_tuning/image_classification_efficientnet_fine_tuning_25_1.png)

### EfficientNet 미세 조정을 위한 팁 {#tips-for-fine-tuning-efficientnet}

동결 해제 레이어에서:

- `BatchNormalization` 레이어는 동결 상태로 유지해야 합니다. ([자세한 내용]({{< relref "/docs/guides/transfer_learning" >}})) 트레이닝 가능한 상태로 전환하면, 동결 해제 후 첫 번째 에포크에서 정확도가 크게 떨어집니다.
- 경우에 따라서는 모든 레이어를 동결 해제하는 대신, 일부 레이어만 동결 해제하는 것이 유리할 수 있습니다. 이렇게 하면 B7과 같은 큰 모델로 이동할 때 미세 조정이 훨씬 빨라집니다.
- 각 블록을 모두 켜거나 꺼야 합니다. 이는 아키텍처에 각 블록의 첫 번째 레이어에서 마지막 레이어까지 shortcut이 포함되어 있기 때문입니다. 블록을 존중하지 않으면 최종 성능도 크게 저하됩니다.

EfficientNet을 활용하기 위한 몇 가지 다른 팁:

- 특히 데이터가 적거나 클래스 수가 적은 작업의 경우, EfficientNet의 변형이 클수록 성능이 향상되지 않습니다. 이러한 경우, 더 큰 EfficientNet 변형을 선택할수록, 하이퍼파라미터를 조정하기가 더 어려워집니다.
- EMA(지수이동평균, Exponential Moving Average)는 EfficientNet을 처음부터 트레이닝하는 데는 매우 유용하지만, 전이 학습에는 그다지 유용하지 않습니다.
- 전이 학습에는 원본 논문에서와 같은 RMSprop 설정을 사용하지 마세요. 전이 학습을 하기에는 모멘텀과 학습률이 너무 높습니다. 사전 트레이닝된 가중치가 쉽게 손상되어 손실이 커질 수 있습니다. 간단한 확인 방법은 손실(카테고리형 교차 엔트로피)이 같은 에포크 이후 log(NUM_CLASSES)보다 현저히 커지는지 확인하는 것입니다. 만약 그렇다면, 초기 학습률/모멘텀이 너무 높다는 뜻입니다.
- 배치 크기가 작을수록 정규화를 효과적으로 제공하기 때문에, 검증 정확도에 도움이 될 수 있습니다.
