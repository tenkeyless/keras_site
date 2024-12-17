---
title: 최신 MLP 모델을 사용한 이미지 분류
linkTitle: 최신 MLP 모델 이미지 분류
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

**{{< t f_author >}}** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)  
**{{< t f_date_created >}}** 2021/05/30  
**{{< t f_last_modified >}}** 2023/08/03  
**{{< t f_description >}}** CIFAR-100 이미지 분류를 위한 MLP-Mixer, FNet 및 gMLP 모델 구현하기.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/mlp_image_classification.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/mlp_image_classification.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

이 예는 이미지 분류를 위한 세 가지 최신 어텐션 프리, 다층 퍼셉트론(MLP) 기반 모델을 구현하며, CIFAR-100 데이터 세트에 대해 시연합니다:

1.  [MLP-Mixer](https://arxiv.org/abs/2105.01601) 모델 - Ilya Tolstikhin et al.가 만든, 두 가지 유형의 MLP를 기반으로 하는 모델
2.  [FNet](https://arxiv.org/abs/2105.03824) 모델 - James Lee-Thorp et al.가 만든, 파라미터화되지 않은 푸리에 변환을 기반으로 하는 모델
3.  [gMLP](https://arxiv.org/abs/2105.08050) 모델 - Hanxiao Liu et al. 등이 만든, 게이팅이 있는 MLP를 기반으로 하는 모델

이 예제의 목적은 하이퍼파라미터가 잘 조정된 데이터 세트에 따라 성능이 달라질 수 있으므로,
이 모델들을 비교하는 것이 아닙니다.
그보다는, 주요 빌딩 블록의 간단한 구현을 보여주기 위한 것입니다.

## 셋업 {#setup}

```python
import numpy as np
import keras
from keras import layers
```

## 데이터 준비 {#prepare-the-data}

```python
num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 1)
x_test shape: (10000, 32, 32, 3) - y_test shape: (10000, 1)
```

{{% /details %}}

## 하이퍼파라미터 구성 {#configure-the-hyperparameters}

```python
weight_decay = 0.0001
batch_size = 128
num_epochs = 1  # 추천 num_epochs = 50
dropout_rate = 0.2
image_size = 64  # 입력 이미지의 크기를 이 크기로 조정합니다.
patch_size = 8  # 입력 이미지에서 추출할 패치의 크기입니다.
num_patches = (image_size // patch_size) ** 2  # 데이터 배열의 크기입니다.
embedding_dim = 256  # 숨겨진 유닛 수입니다.
num_blocks = 4  # 블록 수입니다.

print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")
print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Image size: 64 X 64 = 4096
Patch size: 8 X 8 = 64
Patches per image: 64
Elements per patch (3 channels): 192
```

{{% /details %}}

## 분류 모델 빌드 {#build-a-classification-model}

주어진 처리 블록에 대해 분류기를 빌드하는 메서드를 구현합니다.

```python
def build_classifier(blocks, positional_encoding=False):
    inputs = layers.Input(shape=input_shape)
    # 데이터 보강.
    augmented = data_augmentation(inputs)
    # 패치 생성.
    patches = Patches(patch_size)(augmented)
    # 패치를 인코딩하여 [batch_size, num_patches, embedding_dim] 텐서를 생성.
    x = layers.Dense(units=embedding_dim)(patches)
    if positional_encoding:
        x = x + PositionEmbedding(sequence_length=num_patches)(x)
    # 모듈 블록을 사용하여 x를 처리.
    x = blocks(x)
    # 글로벌 평균 풀링을 적용하여, [batch_size, embedding_dim] 표현 텐서를 생성.
    representation = layers.GlobalAveragePooling1D()(x)
    # 드롭아웃 적용.
    representation = layers.Dropout(rate=dropout_rate)(representation)
    # 로그 출력 계산.
    logits = layers.Dense(num_classes)(representation)
    # Keras 모델 생성.
    return keras.Model(inputs=inputs, outputs=logits)
```

## 실험 정의하기 {#define-an-experiment}

주어진 모델을 컴파일, 트레이닝 및 평가하는 유틸리티 함수를 구현합니다.

```python
def run_experiment(model):
    # 가중치 감쇠가 있는 Adam 옵티마이저를 생성.
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    # 모델 컴파일.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )
    # 학습률 스케줄러 콜백 생성.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    )
    # 조기 중지 콜백 생성.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    # 모델 Fit.
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # 학습 곡선을 그리기 위해 히스토리를 반환.
    return history
```

## 데이터 보강 사용 {#use-data-augmentation}

```python
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# 정규화를 위해 트레이닝 데이터의 평균과 분산을 계산.
data_augmentation.layers[0].adapt(x_train)
```

## 패치 추출을 레이어로서 구현하기 {#implement-patch-extraction-as-a-layer}

```python
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        patches = keras.ops.image.extract_patches(x, self.patch_size)
        batch_size = keras.ops.shape(patches)[0]
        num_patches = keras.ops.shape(patches)[1] * keras.ops.shape(patches)[2]
        patch_dim = keras.ops.shape(patches)[3]
        out = keras.ops.reshape(patches, (batch_size, num_patches, patch_dim))
        return out
```

## 위치 임베딩을 레이어로서 구현하기 {#implement-position-embedding-as-a-layer}

```python
class PositionEmbedding(keras.layers.Layer):
    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, start_index=0):
        shape = keras.ops.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        # trim을 사용하여 입력 시퀀스의 길이와 일치하도록 길이를 조정할 수 있으며,
        # 이는 레이어의 sequence_length보다 작을 수도 있습니다.
        position_embeddings = keras.ops.convert_to_tensor(self.position_embeddings)
        position_embeddings = keras.ops.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return keras.ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape
```

## MLP-Mixer 모델 {#the-mlp-mixer-model}

MLP-Mixer는 MLP(다층 퍼셉트론)에만 기반한 아키텍처로, 두 가지 유형의 MLP 레이어가 포함되어 있습니다:

1.  하나는 이미지 패치에 독립적으로 적용되어, 위치별 특성을 혼합(mix)합니다.
2.  다른 하나는 패치에 걸쳐(채널을 따라) 적용되어, 공간 정보를 혼합(mix)합니다.

이는 Xception 모델과 같은
[깊이 분리형 컨볼루션 기반 모델](https://arxiv.org/abs/1610.02357)과 유사하지만,
두 개의 체인된 Dense 변환, 최대 풀링 없음, 배치 정규화 대신 레이어 정규화라는 차이점이 있습니다.

### MLP-Mixer 모듈 구현하기 {#implement-the-mlp-mixer-module}

```python
class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=hidden_units),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        # 레이어 정규화 적용.
        x = self.normalize(inputs)
        # 입력을 [num_batches, num_patches, hidden_units]에서 [num_batches, hidden_units, num_patches]로 Transpose.
        x_channels = keras.ops.transpose(x, axes=(0, 2, 1))
        # 각 채널에 mlp1을 독립적으로 적용.
        mlp1_outputs = self.mlp1(x_channels)
        # mlp1_outputs을 [num_batches, hidden_dim, num_patches]에서 [num_batches, num_patches, hidden_units]로 Transpose.
        mlp1_outputs = keras.ops.transpose(mlp1_outputs, axes=(0, 2, 1))
        # 스킵 연결 추가.
        x = mlp1_outputs + inputs
        # 레이어 정규화 적용.
        x_patches = self.normalize(x)
        # 각 패치에 mlp2를 독립적으로 적용.
        mlp2_outputs = self.mlp2(x_patches)
        # 스킵 연결 추가.
        x = x + mlp2_outputs
        return x
```

### MLP-Mixer 모델 빌드, 트레이닝 및 평가하기 {#build-train-and-evaluate-the-mlp-mixer-model}

V100 GPU에서 현재 설정으로 모델을 트레이닝하는 데는 에포크 당 약 8초가 소요됩니다.

```python
mlpmixer_blocks = keras.Sequential(
    [MLPMixerLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
)
learning_rate = 0.005
mlpmixer_classifier = build_classifier(mlpmixer_blocks)
history = run_experiment(mlpmixer_classifier)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Test accuracy: 9.76%
Test top 5 accuracy: 30.8%
```

{{% /details %}}

MLP-Mixer 모델은 컨볼루션 및 트랜스포머 기반 모델에 비해 파라미터 수가 훨씬 적기 때문에,
트레이닝 및 서버 계산 비용이 적게 듭니다.

[MLP-Mixer](https://arxiv.org/abs/2105.01601) 논문에서 언급했듯이,
대규모 데이터 세트에 대해 사전 트레이닝할 때 또는 최신 정규화 체계를 사용할 때,
MLP-Mixer는 최신 모델과 경쟁할 수 있는 점수를 얻을 수 있습니다.
임베딩 차원을 늘리고, 믹서 블록 수를 늘리고,
모델을 더 오래 트레이닝하면 더 나은 결과를 얻을 수 있습니다.
입력 이미지의 크기를 늘리고 다른 패치 크기를 사용해 볼 수도 있습니다.

## FNet 모델 {#the-fnet-model}

FNet은 트랜스포머 블록과 유사한 블록을 사용합니다.
하지만, FNet은 트랜스포머 블록의 셀프 어텐션 레이어를
파라미터가 없는 2D 푸리에 변환 레이어로 대체합니다:

1.  패치를 따라 하나의 1D 푸리에 변환이 적용됩니다.
2.  채널을 따라 하나의 1D 푸리에 변환이 적용됩니다.

### FNet 모듈 구현 {#implement-the-fnet-module}

```python
class FNetLayer(layers.Layer):
    def __init__(self, embedding_dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ffn = keras.Sequential(
            [
                layers.Dense(units=embedding_dim, activation="gelu"),
                layers.Dropout(rate=dropout_rate),
                layers.Dense(units=embedding_dim),
            ]
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # 푸리에 변환 적용.
        real_part = inputs
        im_part = keras.ops.zeros_like(inputs)
        x = keras.ops.fft2((real_part, im_part))[0]
        # 스킵 연결 추가.
        x = x + inputs
        # 레이어 정규화 적용.
        x = self.normalize1(x)
        # Feedfowrad 네트워크 적용.
        x_ffn = self.ffn(x)
        # 스킵 연결 추가.
        x = x + x_ffn
        # 레이어 정규화 적용.
        return self.normalize2(x)
```

### FNet 모델 빌드, 트레이닝 및 평가하기 {#build-train-and-evaluate-the-fnet-model}

V100 GPU에서 현재 설정으로 모델을 트레이닝하는 데는 에포크 당 약 8초가 소요됩니다.

```python
fnet_blocks = keras.Sequential(
    [FNetLayer(embedding_dim, dropout_rate) for _ in range(num_blocks)]
)
learning_rate = 0.001
fnet_classifier = build_classifier(fnet_blocks, positional_encoding=True)
history = run_experiment(fnet_classifier)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Test accuracy: 13.82%
Test top 5 accuracy: 36.15%
```

{{% /details %}}

[FNet](https://arxiv.org/abs/2105.03824) 논문에서 볼 수 있듯이,
임베딩 차원을 늘리고, FNet 블록 수를 늘리고,
모델을 더 오래 트레이닝하면 더 나은 결과를 얻을 수 있습니다.
또한 입력 이미지의 크기를 늘리고 다른 패치 크기를 사용해 볼 수도 있습니다.
FNet은 긴 입력에 매우 효율적으로 확장되고,
어텐션 기반 Transformer 모델보다 훨씬 빠르게 실행되며,
경쟁력 있는 정확도 결과를 생성합니다.

## gMLP 모델 {#the-gmlp-model}

gMLP는 공간 게이팅 유닛(SGU, Spatial Gating Unit)이 특징인 MLP 아키텍처입니다.
SGU는 다음과 같이 공간(채널) 차원에 걸쳐 교차 패치 상호 작용을 가능하게 합니다:

1.  패치에 걸쳐(채널을 따라) 선형 투영을 적용하여, 입력을 공간적으로 변환합니다.
2.  입력의 요소별 곱셈과 공간 변환(spatial transformation)을 적용합니다.

### gMLP 모듈 구현 {#implement-the-gmlp-module}

```python
class gMLPLayer(layers.Layer):
    def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel_projection1 = keras.Sequential(
            [
                layers.Dense(units=embedding_dim * 2, activation="gelu"),
                layers.Dropout(rate=dropout_rate),
            ]
        )

        self.channel_projection2 = layers.Dense(units=embedding_dim)

        self.spatial_projection = layers.Dense(
            units=num_patches, bias_initializer="Ones"
        )

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def spatial_gating_unit(self, x):
        # 채널 차원을 따라 x를 분할.
        # 텐서 u와 v는 [batch_size, num_patchs, embedding_dim] 모양이 됩니다.
        u, v = keras.ops.split(x, indices_or_sections=2, axis=2)
        # 레이어 정규화 적용.
        v = self.normalize2(v)
        # spatial 프로젝션 적용.
        v_channels = keras.ops.transpose(v, axes=(0, 2, 1))
        v_projected = self.spatial_projection(v_channels)
        v_projected = keras.ops.transpose(v_projected, axes=(0, 2, 1))
        # 요소 별 곱 적용.
        return u * v_projected

    def call(self, inputs):
        # 레이어 정규화 적용.
        x = self.normalize1(inputs)
        # 첫 번째 채널 프로젝션 적용. x_projected 모양: [batch_size, num_patches, embedding_dim * 2].
        x_projected = self.channel_projection1(x)
        # 공간 게이팅 유닛(spatial gating unit)을 적용. x_spatial 모양: [batch_size, num_patches, embedding_dim].
        x_spatial = self.spatial_gating_unit(x_projected)
        # 두 번째 채널 프로젝션 적용. x_projected 모양: [batch_size, num_patches, embedding_dim].
        x_projected = self.channel_projection2(x_spatial)
        # 스킵 연결 추가.
        return x + x_projected
```

### gMLP 모델 빌드, 트레이닝 및 평가하기 {#build-train-and-evaluate-the-gmlp-model}

V100 GPU에서 현재 설정으로 모델을 트레이닝하는 데는 에포크 당 약 9초가 소요됩니다.

```python
gmlp_blocks = keras.Sequential(
    [gMLPLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
)
learning_rate = 0.003
gmlp_classifier = build_classifier(gmlp_blocks)
history = run_experiment(gmlp_classifier)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Test accuracy: 17.05%
Test top 5 accuracy: 42.57%
```

{{% /details %}}

[gMLP](https://arxiv.org/abs/2105.08050) 논문에서 볼 수 있듯이,
임베딩 크기를 늘리고, gMLP 블록의 수를 늘리고,
모델을 더 오래 트레이닝하면 더 나은 결과를 얻을 수 있습니다.
또한 입력 이미지의 크기를 늘리고 다른 패치 크기를 사용해 볼 수도 있습니다.
이 논문에서는 AutoAugment뿐만 아니라,
MixUp 및 CutMix와 같은 고급 정규화 전략을 사용했다는 점에 유의하세요.
