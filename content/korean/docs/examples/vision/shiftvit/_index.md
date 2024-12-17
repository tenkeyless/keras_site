---
title: 어텐션이 없는 비전 트랜스포머
linkTitle: 어텐션 없는 비전 트랜스포머
toc: true
weight: 18
type: docs
---

{{< keras/original checkedAt="2024-11-20" >}}

**{{< t f_author >}}** [Aritra Roy Gosthipaty](https://twitter.com/ariG23498), [Ritwik Raha](https://twitter.com/ritwik_raha), [Shivalika Singh](https://www.linkedin.com/in/shivalika-singh/)  
**{{< t f_date_created >}}** 2022/02/24  
**{{< t f_last_modified >}}** 2022/10/15  
**{{< t f_description >}}** ShiftViT의 최소 구현입니다.

{{< keras/version v=2 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/shiftvit.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/shiftvit.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

[비전 트랜스포머](https://arxiv.org/abs/2010.11929)(Vision Transformers, ViTs)는 트랜스포머와 컴퓨터 비전(CV)의 교차점에서 연구의 물결을 촉발시켰습니다.

ViT는, Transformer 블록의 Multi-Head Self-Attention 메커니즘 덕분에,
장거리 및 단거리 종속성을 동시에 모델링할 수 있습니다.
많은 연구자들은 ViT의 성공이 순전히 어텐션 레이어에 기인한다고 믿고 있으며,
ViT 모델의 다른 부분에 대해서는 거의 생각하지 않습니다.

학술 논문 [Shift 연산이 Vision Transformer를 만났을 때: 어텐션 메커니즘에 대한 매우 간단한 대안(When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism)](https://arxiv.org/abs/2201.10801)에서
저자는 **NO PARAMETER**의 연산를 도입하여 ViT의 성공을 쉽게 설명할 것을 제안합니다.
그들은 어텐션 연산을 이동 연산으로 바꿉니다.

이 예에서는, 저자의 [공식 구현](https://github.com/microsoft/SPACH/blob/main/models/shiftvit.py)과 근접하게 일치하는 논문을 최소한으로 구현합니다.

이 예에는 TensorFlow 2.9 이상이 필요하며,
다음 명령을 사용하여 설치할 수 있는 TensorFlow Addons도 필요합니다.

```python
!pip install -qq -U tensorflow-addons
```

## 셋업 및 import {#setup-and-imports}

```python
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import pathlib
import glob

# 재현성을 위한 시드 설정
SEED = 42
keras.utils.set_random_seed(SEED)
```

## 하이퍼파라미터 {#hyperparameters}

이는 실험을 위해 선택한 하이퍼파라미터입니다. 자유롭게 조정하시기 바랍니다.

```python
class Config(object):
    # 데이터
    batch_size = 256
    buffer_size = batch_size * 2
    input_shape = (32, 32, 3)
    num_classes = 10

    # 보강
    image_size = 48

    # 아키텍쳐
    patch_size = 4
    projected_dim = 96
    num_shift_blocks_per_stages = [2, 4, 8, 2]
    epsilon = 1e-5
    stochastic_depth_rate = 0.2
    mlp_dropout_rate = 0.2
    num_div = 12
    shift_pixel = 1
    mlp_expand_ratio = 2

    # 옵티마이저
    lr_start = 1e-5
    lr_max = 1e-3
    weight_decay = 1e-4

    # 트레이닝
    epochs = 100

    # 추론
    label_map = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    tf_ds_batch_size = 20


config = Config()
```

## CIFAR-10 데이터 세트 로드 {#load-the-cifar-10-dataset}

우리는 실험에 CIFAR-10 데이터 세트를 사용합니다.

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
(x_train, y_train), (x_val, y_val) = (
    (x_train[:40000], y_train[:40000]),
    (x_train[40000:], y_train[40000:]),
)
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Testing samples: {len(x_test)}")

AUTO = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(config.buffer_size).batch(config.batch_size).prefetch(AUTO)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(config.batch_size).prefetch(AUTO)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(config.batch_size).prefetch(AUTO)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170498071/170498071 [==============================] - 3s 0us/step
Training samples: 40000
Validation samples: 10000
Testing samples: 10000
```

{{% /details %}}

## 데이터 보강 {#data-augmentation}

보강 파이프라인은 다음으로 구성됩니다.

- 크기 재조정 (Rescaling)
- 크기 조정 (Resizing)
- 무작위 자르기 (Random cropping)
- 무작위 수평 뒤집기 (Random horizontal flipping)

_참고_: 이미지 데이터 보강 레이어는 추론 시 데이터 변환을 적용하지 않습니다.
이는 `training=False`로 이러한 레이어를 호출하면 다르게 동작한다는 의미입니다.
자세한 내용은 [문서]({{< relref "/docs/api/layers/preprocessing_layers/image_augmentation" >}})를 참조하세요.

```python
def get_augmentation_model():
    """데이터 보강 모델 빌드."""
    data_augmentation = keras.Sequential(
        [
            layers.Resizing(config.input_shape[0] + 20, config.input_shape[0] + 20),
            layers.RandomCrop(config.image_size, config.image_size),
            layers.RandomFlip("horizontal"),
            layers.Rescaling(1 / 255.0),
        ]
    )
    return data_augmentation
```

## ShiftViT 아키텍쳐 {#the-shiftvit-architecture}

이 섹션에서는, [ShiftViT 논문](https://arxiv.org/abs/2201.10801)에서 제안된 아키텍처를 구축합니다.

![ShiftViT Architecture](/images/examples/vision/shiftvit/CHU40HX.png "그림 1: ShiftViT의 전체 아키텍처. [Source](https://arxiv.org/abs/2201.10801)")

그림 1에 표시된 아키텍처는,
[Swin Transformer: Shifted Windows를 사용하는 Hierarchical Vision Transformer](https://arxiv.org/abs/2103.14030)에서 영감을 받았습니다.
여기서 저자는 4단계로 구성된 모듈식 아키텍처를 제안합니다.
각 단계는 자체 공간 크기에 따라 작동하여, 계층적 아키텍처를 생성합니다.

`HxWx3` 크기의 입력 이미지는 `4x4` 크기의 겹치지 않는 패치로 분할됩니다.
이는 패치화 레이어(patchify layer)를 통해 수행되며,
결과적으로 특성 크기 `48` (`4x4x3`)의 개별 토큰이 생성됩니다.
각 단계는 두 부분으로 구성됩니다.

1. 임베딩 생성 (Embedding Generation)
2. 쌓인 Shift 블록 (Stacked Shift Blocks)

다음에서는 단계와 모듈에 대해 자세히 설명합니다.

_참고_: [공식 구현](https://github.com/microsoft/SPACH/blob/main/models/shiftvit.py)과 비교하여, Keras API에 더 잘 맞도록 일부 주요 구성 요소를 재구성했습니다.

### ShiftViT 블록 {#the-shiftvit-block}

![ShiftViT block](/images/examples/vision/shiftvit/IDe35vo.gif "그림 2: Model에서 Shift Block으로.")

ShiftViT 아키텍처의 각 단계는 그림 2와 같이 Shift Block으로 구성됩니다.

![Shift Vit Block](/images/examples/vision/shiftvit/0q13pLu.png "그림 3: Shift ViT Block. [Source](https://arxiv.org/abs/2201.10801)")

그림 3에 표시된 Shift Block은 다음과 같이 구성됩니다:

1.  Shift Operation
2.  Linear Normalization
3.  MLP 레이어

#### MLP 블록 {#the-mlp-block}

MLP 블록은 densely-connected 레이어 스택으로 설계되었습니다.

```python
class MLP(layers.Layer):
    """각 시프트 블록에 대한 MLP 레이어를 가져옵니다.

    Args:
        mlp_expand_ratio (int): 첫 번째 특성 맵이 확장되는 비율입니다.
        mlp_dropout_rate (float): Dropout 비율입니다.
    """

    def __init__(self, mlp_expand_ratio, mlp_dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.mlp_expand_ratio = mlp_expand_ratio
        self.mlp_dropout_rate = mlp_dropout_rate

    def build(self, input_shape):
        input_channels = input_shape[-1]
        initial_filters = int(self.mlp_expand_ratio * input_channels)

        self.mlp = keras.Sequential(
            [
                layers.Dense(
                    units=initial_filters,
                    activation=tf.nn.gelu,
                ),
                layers.Dropout(rate=self.mlp_dropout_rate),
                layers.Dense(units=input_channels),
                layers.Dropout(rate=self.mlp_dropout_rate),
            ]
        )

    def call(self, x):
        x = self.mlp(x)
        return x
```

#### DropPath 레이어 {#the-droppath-layer}

확률적 깊이(Stochastic depth)는 일련의 레이어를 무작위로 drop 하는 정규화 기법입니다.
추론하는 동안, 레이어는 그대로 유지됩니다.
Dropout과 매우 유사하지만, 레이어 내부에 존재하는 개별 노드가 아닌, 레이어 블록에 대해 작동합니다.

```python
class DropPath(layers.Layer):
    """Stochastic Depth 레이어라고도 하는 Drop Path입니다.

    Refernece:
        - https://keras.io/examples/vision/cct/#stochastic-depth-for-regularization
        - github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_prob = drop_path_prob

    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_path_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
```

#### Block {#block}

이 논문에서 가장 중요한 연산은 **시프트 연산(shift operation)**입니다.
이 섹션에서는, 시프트 연산에 대해 설명하고 저자가 제공한 원본 구현과 비교합니다.

일반적인 특성 맵은 `[N, H, W, C]` 모양을 가진다고 가정합니다.
여기서는 채널의 분할 크기를 결정하는 `num_div` 매개 변수를 선택합니다.
처음 4개의 분할은 왼쪽, 오른쪽, 위, 아래 방향으로 시프트(1픽셀)됩니다.
나머지 분할은 그대로 유지됩니다. 부분 시프트 후 시프트된 채널이 채워지고(padded)
오버플로된 픽셀이 잘립니다(chopped off). 이것으로 부분 시프트 작업이 완료됩니다.

원본 구현에서, 코드는 대략 다음과 같습니다:

```python
out[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  # 왼쪽으로 시프트
out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # 오른쪽으로 시프트
out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # 위쪽으로 시프트
out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # 아래쪽으로 시프트

out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # 시프트 없음
```

TensorFlow에서는 트레이닝 과정 중간에 텐서에 시프트된 채널을 할당하는 것은 불가능합니다.
그래서 다음과 같은 절차를 사용했습니다:

1.  `num_div` 매개 변수를 사용하여, 채널을 분할합니다.
2.  처음 4개의 분할 각각을 선택하고, 각 방향으로 시프트 및 패딩합니다.
3.  시프트 및 패딩 후, 채널을 다시 연결(concatenate)합니다.

![Manim rendered animation for shift operation](/images/examples/vision/shiftvit/PReeULP.gif "그림 4: TensorFlow 스타일의 시프트")

전체 절차는 그림 4에 설명되어 있습니다.

```python
class ShiftViTBlock(layers.Layer):
    """ShiftViT Block 유닛

    Args:
        shift_pixel (int): 이동할 픽셀 수입니다. 기본값은 1입니다.
        mlp_expand_ratio (int): MLP 특성이 확장되는 비율입니다. 기본값은 2입니다.
        mlp_dropout_rate (float): MLP에서 사용되는 dropout 비율입니다.
        num_div (int): 특성 맵 채널의 분할 수입니다. 총, 4/num_div의 채널이 시프트됩니다. 기본값은 12입니다.
        epsilon (float): Epsilon 상수입니다.
        drop_path_prob (float): drop path에 대한 drop 확률입니다.
    """

    def __init__(
        self,
        epsilon,
        drop_path_prob,
        mlp_dropout_rate,
        num_div=12,
        shift_pixel=1,
        mlp_expand_ratio=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shift_pixel = shift_pixel
        self.mlp_expand_ratio = mlp_expand_ratio
        self.mlp_dropout_rate = mlp_dropout_rate
        self.num_div = num_div
        self.epsilon = epsilon
        self.drop_path_prob = drop_path_prob

    def build(self, input_shape):
        self.H = input_shape[1]
        self.W = input_shape[2]
        self.C = input_shape[3]
        self.layer_norm = layers.LayerNormalization(epsilon=self.epsilon)
        self.drop_path = (
            DropPath(drop_path_prob=self.drop_path_prob)
            if self.drop_path_prob > 0.0
            else layers.Activation("linear")
        )
        self.mlp = MLP(
            mlp_expand_ratio=self.mlp_expand_ratio,
            mlp_dropout_rate=self.mlp_dropout_rate,
        )

    def get_shift_pad(self, x, mode):
        """선택한 모드에 따라 채널을 시프트합니다."""
        if mode == "left":
            offset_height = 0
            offset_width = 0
            target_height = 0
            target_width = self.shift_pixel
        elif mode == "right":
            offset_height = 0
            offset_width = self.shift_pixel
            target_height = 0
            target_width = self.shift_pixel
        elif mode == "up":
            offset_height = 0
            offset_width = 0
            target_height = self.shift_pixel
            target_width = 0
        else:
            offset_height = self.shift_pixel
            offset_width = 0
            target_height = self.shift_pixel
            target_width = 0
        crop = tf.image.crop_to_bounding_box(
            x,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=self.H - target_height,
            target_width=self.W - target_width,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=self.H,
            target_width=self.W,
        )
        return shift_pad

    def call(self, x, training=False):
        # 특성 맵 분할
        x_splits = tf.split(x, num_or_size_splits=self.C // self.num_div, axis=-1)

        # 특성 맵 시프트
        x_splits[0] = self.get_shift_pad(x_splits[0], mode="left")
        x_splits[1] = self.get_shift_pad(x_splits[1], mode="right")
        x_splits[2] = self.get_shift_pad(x_splits[2], mode="up")
        x_splits[3] = self.get_shift_pad(x_splits[3], mode="down")

        # 시프트된 특성 맵과 시프트되지 않은 특성 맵 연결(Concatenate)하기
        x = tf.concat(x_splits, axis=-1)

        # residual 연결 추가
        shortcut = x
        x = shortcut + self.drop_path(self.mlp(self.layer_norm(x)), training=training)
        return x
```

### ShiftViT 블록 {#the-shiftvit-blocks}

![Shift Blokcs](/images/examples/vision/shiftvit/FKy5NnD.png "그림 5: 아키텍처의 Shift 블록. [Source](https://arxiv.org/abs/2201.10801)")

아키텍처의 각 단계에는 그림 5와 같이 시프트 블록이 있습니다.
이러한 각 블록에는 (이전 섹션에서 구축한 것처럼) 다양한 수의 stacked ShiftViT 블록이 포함되어 있습니다.

Shift 블록 뒤에는 특성 입력을 축소(scales down)하는 PatchMerging 레이어가 이어집니다.
PatchMerging 레이어는 모델의 피라미드 구조에 도움이 됩니다.

#### PatchMerging 레이어 {#the-patchmerging-layer}

이 레이어는 인접한 두 개의 토큰을 병합합니다.
이 레이어는 특성를 공간적으로 축소하고, 채널별(channel wise)로 피처를 늘리는 데 도움이 됩니다.
Conv2D 레이어를 사용하여 패치를 병합합니다.

```python
class PatchMerging(layers.Layer):
    """Patch Merging 레이어.

    Args:
        epsilon (float): epsilon 상수입니다.
    """

    def __init__(self, epsilon, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        filters = 2 * input_shape[-1]
        self.reduction = layers.Conv2D(
            filters=filters, kernel_size=2, strides=2, padding="same", use_bias=False
        )
        self.layer_norm = layers.LayerNormalization(epsilon=self.epsilon)

    def call(self, x):
        # 특성 맵에 패치 병합 알고리즘 적용하기
        x = self.layer_norm(x)
        x = self.reduction(x)
        return x
```

#### Stacked Shift 블록 {#stacked-shift-blocks}

각 단계에는 논문에서 제안한 대로 다양한 수의 ShiftViT 블록이 쌓이게 됩니다.
이 레이어는 패치 병합 레이어와 함께 스택된 시프트 ViT 블록을 포함하는 generic 레이어입니다.
두 가지 작업(시프트 ViT 블록과 패치 병합)을 결합하는 것은 코드 재사용성을 높이기 위해 선택한 디자인 선택입니다.

```python
# 참고 : 이 레이어는 모델 단계마다 쌓는 깊이가 다릅니다.
class StackedShiftBlocks(layers.Layer):
    """stacked ShiftViT 블록을 포함하는 레이어입니다.

    Args:
        epsilon (float): epsilon 상수입니다.
        mlp_dropout_rate (float): MLP 블록에서 사용되는 dropout 비율입니다.
        num_shift_blocks (int): 이 스테이지의 shift vit 블록 수입니다.
        stochastic_depth_rate (float): 선택한 최대 drop path 비율입니다.
        is_merge (boolean): shift vit 이후, Patch Merge 레이어의 사용을 결정하는 플래그입니다.
        num_div (int): 특성 맵의 채널 division 입니다. 기본값은 12입니다.
        shift_pixel (int): shift 할 픽셀 수입니다. 기본값은 1입니다.
        mlp_expand_ratio (int): MLP의 초기 dense 레이어가 확장되는 비율입니다.
    """

    def __init__(
        self,
        epsilon,
        mlp_dropout_rate,
        num_shift_blocks,
        stochastic_depth_rate,
        is_merge,
        num_div=12,
        shift_pixel=1,
        mlp_expand_ratio=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.mlp_dropout_rate = mlp_dropout_rate
        self.num_shift_blocks = num_shift_blocks
        self.stochastic_depth_rate = stochastic_depth_rate
        self.is_merge = is_merge
        self.num_div = num_div
        self.shift_pixel = shift_pixel
        self.mlp_expand_ratio = mlp_expand_ratio

    def build(self, input_shapes):
        # stochastic depth probabilities을 계산합니다.
        # Reference: https://keras.io/examples/vision/cct/#the-final-cct-model
        dpr = [
            x
            for x in np.linspace(
                start=0, stop=self.stochastic_depth_rate, num=self.num_shift_blocks
            )
        ]

        # 시프트 블록을 ShiftViT 블록 목록으로 빌드합니다.
        self.shift_blocks = list()
        for num in range(self.num_shift_blocks):
            self.shift_blocks.append(
                ShiftViTBlock(
                    num_div=self.num_div,
                    epsilon=self.epsilon,
                    drop_path_prob=dpr[num],
                    mlp_dropout_rate=self.mlp_dropout_rate,
                    shift_pixel=self.shift_pixel,
                    mlp_expand_ratio=self.mlp_expand_ratio,
                )
            )
        if self.is_merge:
            self.patch_merge = PatchMerging(epsilon=self.epsilon)

    def call(self, x, training=False):
        for shift_block in self.shift_blocks:
            x = shift_block(x, training=training)
        if self.is_merge:
            x = self.patch_merge(x)
        return x

    # 커스텀 레이어이므로, 트레이닝 후 모델을 쉽게 저장하고 로드할 수 있도록
    # get_config()를 덮어써야 합니다.
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "mlp_dropout_rate": self.mlp_dropout_rate,
                "num_shift_blocks": self.num_shift_blocks,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "is_merge": self.is_merge,
                "num_div": self.num_div,
                "shift_pixel": self.shift_pixel,
                "mlp_expand_ratio": self.mlp_expand_ratio,
            }
        )
        return config
```

## ShiftViT 모델 {#the-shiftvit-model}

ShiftViT 커스텀 모델을 구축합니다.

```python
class ShiftViTModel(keras.Model):
    """ShiftViT 모델.

    Args:
        data_augmentation (keras.Model): 데이터 보강 모델입니다.
        projected_dim (int): 이미지의 패치가 프로젝션 될 차원입니다.
        patch_size (int): 이미지의 패치 크기입니다.
        num_shift_blocks_per_stages (list[int]): 스테이지 당 모든 shift 블록의 개수 목록입니다.
        epsilon (float): 엡실론 상수.
        mlp_dropout_rate (float): MLP 블록에 사용되는 드롭아웃 비율입니다.
        stochastic_depth_rate (float): 최대 드랍률 확률.
        num_div (int): 특성 맵의 채널 분할 수. 기본값은 12입니다.
        shift_pixel (int): shift 할 픽셀 수. 기본값은 1입니다.
        mlp_expand_ratio (int): 초기 mlp dense 레이어가 확장되는 비율입니다. 기본값은 2입니다.
    """

    def __init__(
        self,
        data_augmentation,
        projected_dim,
        patch_size,
        num_shift_blocks_per_stages,
        epsilon,
        mlp_dropout_rate,
        stochastic_depth_rate,
        num_div=12,
        shift_pixel=1,
        mlp_expand_ratio=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_augmentation = data_augmentation
        self.patch_projection = layers.Conv2D(
            filters=projected_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="same",
        )
        self.stages = list()
        for index, num_shift_blocks in enumerate(num_shift_blocks_per_stages):
            if index == len(num_shift_blocks_per_stages) - 1:
                # 이것은 마지막 단계이므로, 여기서는 패치 병합을 사용하지 마세요.
                is_merge = False
            else:
                is_merge = True
            # 스테이지를 구축하세요.
            self.stages.append(
                StackedShiftBlocks(
                    epsilon=epsilon,
                    mlp_dropout_rate=mlp_dropout_rate,
                    num_shift_blocks=num_shift_blocks,
                    stochastic_depth_rate=stochastic_depth_rate,
                    is_merge=is_merge,
                    num_div=num_div,
                    shift_pixel=shift_pixel,
                    mlp_expand_ratio=mlp_expand_ratio,
                )
            )
        self.global_avg_pool = layers.GlobalAveragePooling2D()

        self.classifier = layers.Dense(config.num_classes)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "data_augmentation": self.data_augmentation,
                "patch_projection": self.patch_projection,
                "stages": self.stages,
                "global_avg_pool": self.global_avg_pool,
                "classifier": self.classifier,
            }
        )
        return config

    def _calculate_loss(self, data, training=False):
        (images, labels) = data

        # 이미지를 보강하세요
        augmented_images = self.data_augmentation(images, training=training)

        # 패치를 만들고 패치를 프로젝션합니다.
        projected_patches = self.patch_projection(augmented_images)

        # 스테이지를 통과하세요
        x = projected_patches
        for stage in self.stages:
            x = stage(x, training=training)

        # 로짓을 얻으세요.
        x = self.global_avg_pool(x)
        logits = self.classifier(x)

        # 손실을 계산하여 반환하세요.
        total_loss = self.compiled_loss(labels, logits)
        return total_loss, labels, logits

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            total_loss, labels, logits = self._calculate_loss(
                data=inputs, training=True
            )

        # 그라디언트를 적용합니다.
        train_vars = [
            self.data_augmentation.trainable_variables,
            self.patch_projection.trainable_variables,
            self.global_avg_pool.trainable_variables,
            self.classifier.trainable_variables,
        ]
        train_vars = train_vars + [stage.trainable_variables for stage in self.stages]

        # 그라디언트를 최적화합니다.
        grads = tape.gradient(total_loss, train_vars)
        trainable_variable_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                trainable_variable_list.append((g, v))
        self.optimizer.apply_gradients(trainable_variable_list)

        # 메트릭 업데이트
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        _, labels, logits = self._calculate_loss(data=data, training=False)

        # 메트릭 업데이트
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}

    def call(self, images):
        augmented_images = self.data_augmentation(images)
        x = self.patch_projection(augmented_images)
        for stage in self.stages:
            x = stage(x, training=False)
        x = self.global_avg_pool(x)
        logits = self.classifier(x)
        return logits
```

## 모델 인스턴스화 {#instantiate-the-model}

```python
model = ShiftViTModel(
    data_augmentation=get_augmentation_model(),
    projected_dim=config.projected_dim,
    patch_size=config.patch_size,
    num_shift_blocks_per_stages=config.num_shift_blocks_per_stages,
    epsilon=config.epsilon,
    mlp_dropout_rate=config.mlp_dropout_rate,
    stochastic_depth_rate=config.stochastic_depth_rate,
    num_div=config.num_div,
    shift_pixel=config.shift_pixel,
    mlp_expand_ratio=config.mlp_expand_ratio,
)
```

## 학습률 스케쥴 {#learning-rate-schedule}

많은 실험에서, 우리는 학습률을 천천히 증가시켜 모델을 워밍업하고,
그런 다음, 학습률을 천천히 감소시켜 모델을 쿨다운하고 싶어합니다.
워밍업 코사인 감쇠(warmup cosine decay)에서,
학습률은 워밍업 단계에서 선형적으로 증가한 다음, 코사인 감소로 감쇠합니다.

```python
# 일부 코드는 다음에서 가져왔습니다.
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    """워밍업 코사인 감쇠 스케쥴을 사용하는 LearningRateSchedule입니다."""

    def __init__(self, lr_start, lr_max, warmup_steps, total_steps):
        """
        Args:
            lr_start: 초기 학습률
            lr_max: 워밍업 단계에서 lr이 증가해야 하는 최대 학습률
            warmup_steps: 모델이 워밍업하는 단계 수
            total_steps: 모델 트레이닝을 위한 총 단계 수
        """
        super().__init__()
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        # 총 단계 수가 워밍업 단계 수보다 작으면,
        # 값 오류를 throw합니다.
        if self.total_steps < self.warmup_steps:
            raise ValueError(
                f"Total number of steps {self.total_steps} must be"
                + f"larger or equal to warmup steps {self.warmup_steps}."
            )

        # `cos_annealed_lr`은 초기 단계에서 워밍업 단계까지 1로 증가하는 그래프입니다.
        # 그 후, 이 그래프는 최종 단계 마크에서 -1로 감소합니다.
        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        )

        # `cos_annealed_lr` 그래프의 평균을 1로 옮깁니다.
        # 이제 그래프가 0에서 2로 이동합니다.
        # 그래프를 0.5로 정규화하여, 이제 0에서 1로 이동합니다.
        # 정규화된 그래프를 `lr_max`로 조정하여, 0에서 `lr_max`로 이동합니다.
        learning_rate = 0.5 * self.lr_max * (1 + cos_annealed_lr)

        # warmup_steps가 0보다 큰지 확인합니다.
        if self.warmup_steps > 0:
            # lr_max가 lr_start보다 큰지 확인합니다. 그렇지 않으면, 값 오류를 throw합니다.
            if self.lr_max < self.lr_start:
                raise ValueError(
                    f"lr_start {self.lr_start} must be smaller or"
                    + f"equal to lr_max {self.lr_max}."
                )

            # warumup 스케쥴에서 학습률이 증가해야 하는 기울기를 계산합니다.
            # 기울기 공식은 m = ((b-a)/steps)입니다.
            slope = (self.lr_max - self.lr_start) / self.warmup_steps

            # 직선 공식(y = mx+c)을 이용하여 워밍업 스케쥴을 빌드합니다.
            warmup_rate = slope * tf.cast(step, tf.float32) + self.lr_start

            # 현재 단계가 워밍업 단계보다 작으면, 선 그래프를 구합니다.
            # 현재 단계가 워밍업 단계보다 크면, 스케일된 코사인 그래프를 구합니다.
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )

        # 현재 단계가 총 단계보다 큰 경우 0을 반환하고, 그렇지 않으면 계산된 그래프를 반환합니다.
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

    def get_config(self):
        config = {
            "lr_start": self.lr_start,
            "lr_max": self.lr_max,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }
        return config
```

## 모델 컴파일 및 트레이닝 {#compile-and-train-the-model}

```python
# 모델 저장 시 입력 모양을 사용할 수 있도록, 샘플 데이터를 모델에 전달합니다.
sample_ds, _ = next(iter(train_ds))
model(sample_ds, training=False)

# 트레이닝에 필요한 총 스텝 수를 계산합니다.
total_steps = int((len(x_train) / config.batch_size) * config.epochs)

# 워밍업을 위한 스텝 수를 계산하세요.
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps * warmup_epoch_percentage)

# 워밍업 코사인 스케쥴을 초기화합니다.
scheduled_lrs = WarmUpCosine(
    lr_start=1e-5,
    lr_max=1e-3,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
)

# 옵티마이저를 얻습니다.
optimizer = tfa.optimizers.AdamW(
    learning_rate=scheduled_lrs, weight_decay=config.weight_decay
)

# 모델을 컴파일하고 사전 트레이닝합니다.
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)

# 모델 트레이닝
history = model.fit(
    train_ds,
    epochs=config.epochs,
    validation_data=val_ds,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode="auto",
        )
    ],
)

# 테스트 데이터 세트로 모델을 평가합니다.
print("TESTING")
loss, acc_top1, acc_top5 = model.evaluate(test_ds)
print(f"Loss: {loss:0.2f}")
print(f"Top 1 test accuracy: {acc_top1*100:0.2f}%")
print(f"Top 5 test accuracy: {acc_top5*100:0.2f}%")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/100
157/157 [==============================] - 72s 332ms/step - loss: 2.3844 - accuracy: 0.1444 - top-5-accuracy: 0.6051 - val_loss: 2.0984 - val_accuracy: 0.2610 - val_top-5-accuracy: 0.7638
Epoch 2/100
157/157 [==============================] - 49s 314ms/step - loss: 1.9457 - accuracy: 0.2893 - top-5-accuracy: 0.8103 - val_loss: 1.9459 - val_accuracy: 0.3356 - val_top-5-accuracy: 0.8614
Epoch 3/100
157/157 [==============================] - 50s 316ms/step - loss: 1.7093 - accuracy: 0.3810 - top-5-accuracy: 0.8761 - val_loss: 1.5349 - val_accuracy: 0.4585 - val_top-5-accuracy: 0.9045
Epoch 4/100
157/157 [==============================] - 49s 315ms/step - loss: 1.5473 - accuracy: 0.4374 - top-5-accuracy: 0.9090 - val_loss: 1.4257 - val_accuracy: 0.4862 - val_top-5-accuracy: 0.9298
Epoch 5/100
157/157 [==============================] - 50s 316ms/step - loss: 1.4316 - accuracy: 0.4816 - top-5-accuracy: 0.9243 - val_loss: 1.4032 - val_accuracy: 0.5092 - val_top-5-accuracy: 0.9362
Epoch 6/100
157/157 [==============================] - 50s 316ms/step - loss: 1.3588 - accuracy: 0.5131 - top-5-accuracy: 0.9333 - val_loss: 1.2893 - val_accuracy: 0.5411 - val_top-5-accuracy: 0.9457
Epoch 7/100
157/157 [==============================] - 50s 316ms/step - loss: 1.2894 - accuracy: 0.5385 - top-5-accuracy: 0.9410 - val_loss: 1.2922 - val_accuracy: 0.5416 - val_top-5-accuracy: 0.9432
Epoch 8/100
157/157 [==============================] - 49s 315ms/step - loss: 1.2388 - accuracy: 0.5568 - top-5-accuracy: 0.9468 - val_loss: 1.2100 - val_accuracy: 0.5733 - val_top-5-accuracy: 0.9545
Epoch 9/100
157/157 [==============================] - 49s 315ms/step - loss: 1.2043 - accuracy: 0.5698 - top-5-accuracy: 0.9491 - val_loss: 1.2166 - val_accuracy: 0.5675 - val_top-5-accuracy: 0.9520
Epoch 10/100
157/157 [==============================] - 49s 315ms/step - loss: 1.1694 - accuracy: 0.5861 - top-5-accuracy: 0.9528 - val_loss: 1.1738 - val_accuracy: 0.5883 - val_top-5-accuracy: 0.9541
Epoch 11/100
157/157 [==============================] - 50s 316ms/step - loss: 1.1290 - accuracy: 0.5994 - top-5-accuracy: 0.9575 - val_loss: 1.1161 - val_accuracy: 0.6064 - val_top-5-accuracy: 0.9618
Epoch 12/100
157/157 [==============================] - 50s 316ms/step - loss: 1.0861 - accuracy: 0.6157 - top-5-accuracy: 0.9602 - val_loss: 1.1220 - val_accuracy: 0.6133 - val_top-5-accuracy: 0.9576
Epoch 13/100
157/157 [==============================] - 49s 315ms/step - loss: 1.0766 - accuracy: 0.6178 - top-5-accuracy: 0.9612 - val_loss: 1.0108 - val_accuracy: 0.6402 - val_top-5-accuracy: 0.9681
Epoch 14/100
157/157 [==============================] - 49s 315ms/step - loss: 1.0179 - accuracy: 0.6416 - top-5-accuracy: 0.9658 - val_loss: 1.0196 - val_accuracy: 0.6405 - val_top-5-accuracy: 0.9667
Epoch 15/100
157/157 [==============================] - 50s 316ms/step - loss: 1.0028 - accuracy: 0.6470 - top-5-accuracy: 0.9678 - val_loss: 1.0113 - val_accuracy: 0.6415 - val_top-5-accuracy: 0.9672
Epoch 16/100
157/157 [==============================] - 50s 316ms/step - loss: 0.9613 - accuracy: 0.6611 - top-5-accuracy: 0.9710 - val_loss: 1.0516 - val_accuracy: 0.6406 - val_top-5-accuracy: 0.9596
Epoch 17/100
157/157 [==============================] - 50s 316ms/step - loss: 0.9262 - accuracy: 0.6740 - top-5-accuracy: 0.9729 - val_loss: 0.9010 - val_accuracy: 0.6844 - val_top-5-accuracy: 0.9750
Epoch 18/100
157/157 [==============================] - 50s 316ms/step - loss: 0.8768 - accuracy: 0.6916 - top-5-accuracy: 0.9769 - val_loss: 0.8862 - val_accuracy: 0.6908 - val_top-5-accuracy: 0.9767
Epoch 19/100
157/157 [==============================] - 49s 315ms/step - loss: 0.8595 - accuracy: 0.6984 - top-5-accuracy: 0.9768 - val_loss: 0.8732 - val_accuracy: 0.6982 - val_top-5-accuracy: 0.9738
Epoch 20/100
157/157 [==============================] - 50s 317ms/step - loss: 0.8252 - accuracy: 0.7103 - top-5-accuracy: 0.9793 - val_loss: 0.9330 - val_accuracy: 0.6745 - val_top-5-accuracy: 0.9718
Epoch 21/100
157/157 [==============================] - 51s 322ms/step - loss: 0.8003 - accuracy: 0.7180 - top-5-accuracy: 0.9814 - val_loss: 0.8912 - val_accuracy: 0.6948 - val_top-5-accuracy: 0.9728
Epoch 22/100
157/157 [==============================] - 51s 326ms/step - loss: 0.7651 - accuracy: 0.7317 - top-5-accuracy: 0.9829 - val_loss: 0.7894 - val_accuracy: 0.7277 - val_top-5-accuracy: 0.9791
Epoch 23/100
157/157 [==============================] - 52s 328ms/step - loss: 0.7372 - accuracy: 0.7415 - top-5-accuracy: 0.9843 - val_loss: 0.7752 - val_accuracy: 0.7284 - val_top-5-accuracy: 0.9804
Epoch 24/100
157/157 [==============================] - 51s 327ms/step - loss: 0.7324 - accuracy: 0.7423 - top-5-accuracy: 0.9852 - val_loss: 0.7949 - val_accuracy: 0.7340 - val_top-5-accuracy: 0.9792
Epoch 25/100
157/157 [==============================] - 51s 323ms/step - loss: 0.7051 - accuracy: 0.7512 - top-5-accuracy: 0.9858 - val_loss: 0.7967 - val_accuracy: 0.7280 - val_top-5-accuracy: 0.9787
Epoch 26/100
157/157 [==============================] - 51s 323ms/step - loss: 0.6832 - accuracy: 0.7577 - top-5-accuracy: 0.9870 - val_loss: 0.7840 - val_accuracy: 0.7322 - val_top-5-accuracy: 0.9807
Epoch 27/100
157/157 [==============================] - 51s 322ms/step - loss: 0.6609 - accuracy: 0.7654 - top-5-accuracy: 0.9877 - val_loss: 0.7447 - val_accuracy: 0.7434 - val_top-5-accuracy: 0.9816
Epoch 28/100
157/157 [==============================] - 50s 319ms/step - loss: 0.6495 - accuracy: 0.7724 - top-5-accuracy: 0.9883 - val_loss: 0.7885 - val_accuracy: 0.7280 - val_top-5-accuracy: 0.9817
Epoch 29/100
157/157 [==============================] - 50s 317ms/step - loss: 0.6491 - accuracy: 0.7707 - top-5-accuracy: 0.9885 - val_loss: 0.7539 - val_accuracy: 0.7458 - val_top-5-accuracy: 0.9821
Epoch 30/100
157/157 [==============================] - 50s 317ms/step - loss: 0.6213 - accuracy: 0.7823 - top-5-accuracy: 0.9888 - val_loss: 0.7571 - val_accuracy: 0.7470 - val_top-5-accuracy: 0.9815
Epoch 31/100
157/157 [==============================] - 50s 318ms/step - loss: 0.5976 - accuracy: 0.7902 - top-5-accuracy: 0.9906 - val_loss: 0.7430 - val_accuracy: 0.7508 - val_top-5-accuracy: 0.9817
Epoch 32/100
157/157 [==============================] - 50s 318ms/step - loss: 0.5932 - accuracy: 0.7898 - top-5-accuracy: 0.9910 - val_loss: 0.7545 - val_accuracy: 0.7469 - val_top-5-accuracy: 0.9793
Epoch 33/100
157/157 [==============================] - 50s 318ms/step - loss: 0.5977 - accuracy: 0.7850 - top-5-accuracy: 0.9913 - val_loss: 0.7200 - val_accuracy: 0.7569 - val_top-5-accuracy: 0.9830
Epoch 34/100
157/157 [==============================] - 50s 317ms/step - loss: 0.5552 - accuracy: 0.8041 - top-5-accuracy: 0.9920 - val_loss: 0.7377 - val_accuracy: 0.7552 - val_top-5-accuracy: 0.9818
Epoch 35/100
157/157 [==============================] - 50s 319ms/step - loss: 0.5509 - accuracy: 0.8056 - top-5-accuracy: 0.9921 - val_loss: 0.8125 - val_accuracy: 0.7331 - val_top-5-accuracy: 0.9782
Epoch 36/100
157/157 [==============================] - 50s 317ms/step - loss: 0.5296 - accuracy: 0.8116 - top-5-accuracy: 0.9933 - val_loss: 0.6900 - val_accuracy: 0.7680 - val_top-5-accuracy: 0.9849
Epoch 37/100
157/157 [==============================] - 50s 316ms/step - loss: 0.5151 - accuracy: 0.8170 - top-5-accuracy: 0.9941 - val_loss: 0.7275 - val_accuracy: 0.7610 - val_top-5-accuracy: 0.9841
Epoch 38/100
157/157 [==============================] - 50s 317ms/step - loss: 0.5069 - accuracy: 0.8217 - top-5-accuracy: 0.9936 - val_loss: 0.7067 - val_accuracy: 0.7703 - val_top-5-accuracy: 0.9835
Epoch 39/100
157/157 [==============================] - 50s 318ms/step - loss: 0.4771 - accuracy: 0.8304 - top-5-accuracy: 0.9945 - val_loss: 0.7110 - val_accuracy: 0.7668 - val_top-5-accuracy: 0.9836
Epoch 40/100
157/157 [==============================] - 50s 317ms/step - loss: 0.4675 - accuracy: 0.8350 - top-5-accuracy: 0.9956 - val_loss: 0.7130 - val_accuracy: 0.7688 - val_top-5-accuracy: 0.9829
Epoch 41/100
157/157 [==============================] - 50s 319ms/step - loss: 0.4586 - accuracy: 0.8382 - top-5-accuracy: 0.9959 - val_loss: 0.7331 - val_accuracy: 0.7598 - val_top-5-accuracy: 0.9806
Epoch 42/100
157/157 [==============================] - 50s 318ms/step - loss: 0.4558 - accuracy: 0.8380 - top-5-accuracy: 0.9959 - val_loss: 0.7187 - val_accuracy: 0.7722 - val_top-5-accuracy: 0.9832
Epoch 43/100
157/157 [==============================] - 50s 320ms/step - loss: 0.4356 - accuracy: 0.8450 - top-5-accuracy: 0.9958 - val_loss: 0.7162 - val_accuracy: 0.7693 - val_top-5-accuracy: 0.9850
Epoch 44/100
157/157 [==============================] - 49s 314ms/step - loss: 0.4425 - accuracy: 0.8433 - top-5-accuracy: 0.9958 - val_loss: 0.7061 - val_accuracy: 0.7698 - val_top-5-accuracy: 0.9853
Epoch 45/100
157/157 [==============================] - 49s 314ms/step - loss: 0.4072 - accuracy: 0.8551 - top-5-accuracy: 0.9967 - val_loss: 0.7025 - val_accuracy: 0.7820 - val_top-5-accuracy: 0.9848
Epoch 46/100
157/157 [==============================] - 49s 314ms/step - loss: 0.3865 - accuracy: 0.8644 - top-5-accuracy: 0.9970 - val_loss: 0.7178 - val_accuracy: 0.7740 - val_top-5-accuracy: 0.9844
Epoch 47/100
157/157 [==============================] - 49s 313ms/step - loss: 0.3718 - accuracy: 0.8694 - top-5-accuracy: 0.9973 - val_loss: 0.7216 - val_accuracy: 0.7768 - val_top-5-accuracy: 0.9828
Epoch 48/100
157/157 [==============================] - 49s 314ms/step - loss: 0.3733 - accuracy: 0.8673 - top-5-accuracy: 0.9970 - val_loss: 0.7440 - val_accuracy: 0.7713 - val_top-5-accuracy: 0.9841
Epoch 49/100
157/157 [==============================] - 49s 313ms/step - loss: 0.3531 - accuracy: 0.8741 - top-5-accuracy: 0.9979 - val_loss: 0.7220 - val_accuracy: 0.7738 - val_top-5-accuracy: 0.9848
Epoch 50/100
157/157 [==============================] - 49s 314ms/step - loss: 0.3502 - accuracy: 0.8738 - top-5-accuracy: 0.9980 - val_loss: 0.7245 - val_accuracy: 0.7734 - val_top-5-accuracy: 0.9836
TESTING
40/40 [==============================] - 2s 56ms/step - loss: 0.7336 - accuracy: 0.7638 - top-5-accuracy: 0.9855
Loss: 0.73
Top 1 test accuracy: 76.38%
Top 5 test accuracy: 98.55%
```

{{% /details %}}

## 트레이닝 한 모델 저장 {#save-trained-model}

Subclassing으로 모델을 만들었으므로, HDF5 형식으로 모델을 저장할 수 없습니다.

TF SavedModel 형식으로만 저장할 수 있습니다. 일반적으로 모델을 저장하는 데 권장되는 형식이기도 합니다.

```python
model.save("ShiftViT")
```

## 모델 추론 {#model-inference}

**추론을 위한 샘플 데이터 다운로드**

```python
!wget -q 'https://tinyurl.com/2p9483sw' -O inference_set.zip
!unzip -q inference_set.zip
```

**저장된 모델 로드**

```python
# 모델이 저장될 때 커스텀 객체는 포함되지 않습니다.
# 로딩 시, 이러한 객체는 모델 재구성을 위해 전달되어야 합니다.
saved_model = tf.keras.models.load_model(
    "ShiftViT",
    custom_objects={"WarmUpCosine": WarmUpCosine, "AdamW": tfa.optimizers.AdamW},
)
```

**추론을 위한 유틸리티 함수**

```python
def process_image(img_path):
    # 문자열 경로에서 이미지 파일을 읽습니다.
    img = tf.io.read_file(img_path)

    # jpeg를 uint8 텐서로 디코딩
    img = tf.io.decode_jpeg(img, channels=3)

    # 모델에서 허용하는 입력 크기에 맞게 이미지 크기를 조정합니다.
    # `resize()`에 전달된 입력의 dtype을 보존하기 위해, `method`를 `nearest`로 사용합니다.
    img = tf.image.resize(
        img, [config.input_shape[0], config.input_shape[1]], method="nearest"
    )
    return img


def create_tf_dataset(image_dir):
    data_dir = pathlib.Path(image_dir)

    # 이미지 디렉토리를 사용하여, tf.data 데이터세트 생성
    predict_ds = tf.data.Dataset.list_files(str(data_dir / "*.jpg"), shuffle=False)

    # map을 사용하여 문자열 경로를 uint8 이미지 텐서로 변환하고,
    # `num_parallel_calls'를 설정하면 여러 이미지를 병렬로 처리하는 데 도움이 됩니다.
    predict_ds = predict_ds.map(process_image, num_parallel_calls=AUTO)

    # 더 나은 대기 시간과 처리량을 위해 Prefetch Dataset을 만듭니다.
    predict_ds = predict_ds.batch(config.tf_ds_batch_size).prefetch(AUTO)
    return predict_ds


def predict(predict_ds):
    # ShiftViT 모델은 로짓(정규화되지 않은 예측)을 반환합니다.
    logits = saved_model.predict(predict_ds)

    # softmax()를 호출하여 예측을 정규화합니다.
    probabilities = tf.nn.softmax(logits)
    return probabilities


def get_predicted_class(probabilities):
    pred_label = np.argmax(probabilities)
    predicted_class = config.label_map[pred_label]
    return predicted_class


def get_confidence_scores(probabilities):
    # 확률 점수의 인덱스를 내림차순으로 정렬합니다.
    labels = np.argsort(probabilities)[::-1]
    confidences = {
        config.label_map[label]: np.round((probabilities[label]) * 100, 2)
        for label in labels
    }
    return confidences
```

**예측을 얻으세요**

```python
img_dir = "inference_set"
predict_ds = create_tf_dataset(img_dir)
probabilities = predict(predict_ds)
print(f"probabilities: {probabilities[0]}")
confidences = get_confidence_scores(probabilities[0])
print(confidences)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
1/1 [==============================] - 2s 2s/step
probabilities: [8.7329084e-01 1.3162658e-03 6.1781306e-05 1.9132349e-05 4.4482469e-05
 1.8182898e-06 2.2834571e-05 1.1466043e-05 1.2504059e-01 1.9084632e-04]
{'airplane': 87.33, 'ship': 12.5, 'automobile': 0.13, 'truck': 0.02, 'bird': 0.01, 'deer': 0.0, 'frog': 0.0, 'cat': 0.0, 'horse': 0.0, 'dog': 0.0}
```

{{% /details %}}

**예측 보기**

```python
plt.figure(figsize=(10, 10))
for images in predict_ds:
    for i in range(min(6, probabilities.shape[0])):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predicted_class = get_predicted_class(probabilities[i])
        plt.title(predicted_class)
        plt.axis("off")
```

![png](/images/examples/vision/shiftvit/shiftvit_46_0.png)

## 결론 {#conclusion}

이 논문의 가장 큰 기여는 새로운 아키텍처가 아니라,
어텐션 없이 트레이닝된 계층적 ViT가 매우 좋은 성능을 낼 수 있다는 아이디어입니다.
이는 어텐션이 ViT의 성능에 얼마나 필수적인지에 대한 의문을 제기합니다.

호기심이 많은 사람이라면, 어텐션에 기반한 새로운 아키텍처를 제공하기보다는,
ViT의 트레이닝 패러다임과 아키텍처 세부 사항에 더 많은 주의를 기울이는,
[ConvNexT](https://arxiv.org/abs/2201.03545) 논문을 읽어보는 것이 좋습니다.

Acknowledgements:

- 이 프로젝트를 완료하는 데 도움이 되는 리소스를 제공해 준 [PyImageSearch](https://pyimagesearch.com)에 감사드립니다.
- GPU 크레딧을 제공해 준 [JarvisLabs.ai](https://jarvislabs.ai/)에 감사드립니다.
- manim 라이브러리를 제공해 준 [Manim Community](https://www.manim.community/)에 감사드립니다.
- 학습률 스케쥴에 도움을 준 [Puja Roychowdhury](https://twitter.com/pleb_talks)에게 개인적으로 감사의 말씀을 전합니다.

**HuggingFace에서 사용 가능한 예제**

|                                                           트레이닝된 모델                                                            |                                                                    데모                                                                     |
| :----------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-ShiftViT-brightgreen)](https://huggingface.co/keras-io/shiftvit) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Space-ShiftViT-brightgreen)](https://huggingface.co/spaces/keras-io/shiftvit) |
