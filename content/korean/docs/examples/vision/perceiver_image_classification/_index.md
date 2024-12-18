---
title: Perceiver로 이미지 분류
linkTitle: Perceiver 이미지 분류
toc: true
weight: 13
type: docs
---

{{< keras/original checkedAt="2024-11-20" >}}

**{{< t f_author >}}** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)  
**{{< t f_date_created >}}** 2021/04/30  
**{{< t f_last_modified >}}** 2023/12/30  
**{{< t f_description >}}** 이미지 분류를 위한 Perceiver 모델 구현하기.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/perceiver_image_classification.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/perceiver_image_classification.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

이 예는 이미지 분류를 위해 Andrew Jaegle et al.이 개발한
[Perceiver: 반복적 어텐션이 있는 일반적 Perception(Perceiver: General Perception with Iterative Attention)](https://arxiv.org/abs/2103.03206) 모델을 구현하고,
CIFAR-100 데이터 세트에 대해 이를 시연합니다.

Perceiver 모델은 비대칭 어텐션 메커니즘을 활용하여
입력을 반복적으로 증류하여 타이트한 잠재 병목 현상을 해결함으로써,
매우 큰 입력을 처리할 수 있도록 확장할 수 있습니다.

다시 말해, 입력 데이터 배열(예: 이미지)에 `M`개의 요소(즉, 패치)가 있고
여기서 `M`이 크다고 가정해 보겠습니다. 표준 Transformer 모델에서는,
`M` 요소에 대해 셀프 어텐션 연산이 수행됩니다. 이 연산의 복잡도는 `O(M^2)`입니다.
그러나, Perceiver 모델은 `N << M`인 `N` 크기의 요소의 잠재 배열을 생성하고,
두 가지 연산을 반복적으로 수행합니다:

1.  잠재 배열과 데이터 배열 사이의 교차 어텐션 트랜스포머(Cross-attention Transformer) - 이 작업의 복잡도는 `O(M.N)`입니다.
2.  잠재 배열에 대한 셀프 어텐션 트랜스포머(Self-attention Transformer) - 이 작업의 복잡도는 `O(N^2)`입니다.

이 예제에는 Keras 3.0 이상이 필요합니다.

## 셋업 {#setup}

```python
import keras
from keras import layers, activations, ops
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
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 64
num_epochs = 2  # 실제로는 50 에포크를 사용해야 합니다!
dropout_rate = 0.2
image_size = 64  # 입력 이미지의 크기를 이 크기로 조정합니다.
patch_size = 2  # 입력 이미지에서 추출할 패치의 크기입니다.
num_patches = (image_size // patch_size) ** 2  # 데이터 배열의 크기입니다.
latent_dim = 256  # 잠재 배열의 크기입니다.
projection_dim = 256  # 데이터 및 잠재 배열에 각 요소의 임베딩 크기입니다.
num_heads = 8  # 트랜스포머 헤드 수입니다.
ffn_units = [
    projection_dim,
    projection_dim,
]  # 트랜스포머 피드포워드 네트워크의 크기.
num_transformer_blocks = 4
num_iterations = 2  # 크로스 어텐션 및 트랜스포머 모듈의 반복.
classifier_units = [
    projection_dim,
    num_classes,
]  # 최종 분류기의 Feedforward 네트워크 크기입니다.

print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")
print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")
print(f"Latent array shape: {latent_dim} X {projection_dim}")
print(f"Data array shape: {num_patches} X {projection_dim}")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Image size: 64 X 64 = 4096
Patch size: 2 X 2 = 4
Patches per image: 1024
Elements per patch (3 channels): 12
Latent array shape: 256 X 256
Data array shape: 1024 X 256
```

{{% /details %}}

각 픽셀을 데이터 배열의 개별 입력으로 사용하려면, `patch_size`를 1로 설정하세요.

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
# 정규화를 위해 트레이닝 데이터의 평균과 분산을 계산합니다.
data_augmentation.layers[0].adapt(x_train)
```

## 피드포워드 네트워크(FFN, Feedforward network) 구현 {#implement-feedforward-network-ffn}

```python
def create_ffn(hidden_units, dropout_rate):
    ffn_layers = []
    for units in hidden_units[:-1]:
        ffn_layers.append(layers.Dense(units, activation=activations.gelu))

    ffn_layers.append(layers.Dense(units=hidden_units[-1]))
    ffn_layers.append(layers.Dropout(dropout_rate))

    ffn = keras.Sequential(ffn_layers)
    return ffn
```

## 레이어로 패치 생성 구현 {#implement-patch-creation-as-a-layer}

```python
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = ops.shape(images)[0]
        patches = ops.image.extract_patches(
            image=images,
            size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            dilation_rate=1,
            padding="valid",
        )
        patch_dims = patches.shape[-1]
        patches = ops.reshape(patches, [batch_size, -1, patch_dims])
        return patches
```

## 패치 인코딩 레이어 구현 {#implement-the-patch-encoding-layer}

`PatchEncoder` 레이어는 패치를 `latent_dim` 크기의 벡터로 프로젝션하여 선형적으로 변환합니다.
또한, 프로젝션된 벡터에 학습 가능한 위치 임베딩을 추가합니다.

원본 Perceiver 논문에서는 Fourier 특성 위치 인코딩을 사용한다는 점에 유의하세요.

```python
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = ops.arange(start=0, stop=self.num_patches, step=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded
```

## Perceiver 모델 빌드 {#build-the-perceiver-model}

Perceiver는 크로스 어텐션 모듈과
셀프 어텐션이 있는 표준 트랜스포머의 두 가지 모듈로 구성됩니다.

### 크로스 어텐션 모듈 {#cross-attention-module}

크로스 어텐션은 `(latent_dim, projection_dim)` 잠재 배열과
`(data_dim, projection_dim)` 데이터 배열을 입력으로 받아,
`(latent_dim, projection_dim)` 잠재 배열을 출력으로 생성합니다.
크로스 어텐션을 적용하기 위해, `query` 벡터는 잠재 배열에서 생성되고,
`key` 및 `value` 벡터는 인코딩된 이미지에서 생성됩니다.

이 예제에서 데이터 배열은 이미지이며,
여기서 `data_dim`은 `num_patches`로 설정되어 있습니다.

```python
def create_cross_attention_module(
    latent_dim, data_dim, projection_dim, ffn_units, dropout_rate
):
    inputs = {
        # 모양 [1, latent_dim, projection_dim]의 입력으로 잠재 배열을 받습니다.
        "latent_array": layers.Input(
            shape=(latent_dim, projection_dim), name="latent_array"
        ),
        # data_array(인코딩된 이미지)을 모양 [batch_size, data_dim, projection_dim]의 입력으로 받습니다.
        "data_array": layers.Input(shape=(data_dim, projection_dim), name="data_array"),
    }

    # 입력에 레이어 norm 적용
    latent_array = layers.LayerNormalization(epsilon=1e-6)(inputs["latent_array"])
    data_array = layers.LayerNormalization(epsilon=1e-6)(inputs["data_array"])

    # 쿼리 텐서 생성: [1, latent_dim, projection_dim].
    query = layers.Dense(units=projection_dim)(latent_array)
    # 키 텐서 생성: [batch_size, data_dim, projection_dim].
    key = layers.Dense(units=projection_dim)(data_array)
    # 값 텐서 생성: [batch_size, data_dim, projection_dim].
    value = layers.Dense(units=projection_dim)(data_array)

    # 크로스 어텐션 출력 생성: [batch_size, latent_dim, projection_dim].
    attention_output = layers.Attention(use_scale=True, dropout=0.1)(
        [query, key, value], return_attention_scores=False
    )
    # 스킵 연결 1.
    attention_output = layers.Add()([attention_output, latent_array])

    # 레이어 norm 적용.
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    # Feedforward 네트워크 적용.
    ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
    outputs = ffn(attention_output)
    # 스킵 연결 2.
    outputs = layers.Add()([outputs, attention_output])

    # Keras 모델 생성.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
```

### 트랜스포머 모듈 {#transformer-module}

트랜스포머는 크로스 어텐션 모듈의 출력 잠재 벡터를 입력으로 예상하고,
`latent_dim` 요소에 멀티 헤드 셀프 어텐션을 적용한 다음,
피드포워드 네트워크를 통해 또다른 `(latent_dim, projection_dim)` 잠재 배열을 생성합니다.

```python
def create_transformer_module(
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    ffn_units,
    dropout_rate,
):
    # input_shape: [1, latent_dim, projection_dim]
    inputs = layers.Input(shape=(latent_dim, projection_dim))

    x0 = inputs
    # 트랜스포머 블록의 여러 레이어를 생성합니다.
    for _ in range(num_transformer_blocks):
        # 레이어 정규화 1 적용.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x0)
        # 멀티 헤드 셀프 어텐션 레이어를 만듭니다.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # 스킵 연결 1.
        x2 = layers.Add()([attention_output, x0])
        # 레이어 정규화 2 적용.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # Feedforward 네트워크 적용.
        ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
        x3 = ffn(x3)
        # 스킵 연결 2.
        x0 = layers.Add()([x3, x2])

    # Keras 모델 생성.
    model = keras.Model(inputs=inputs, outputs=x0)
    return model
```

### Perceiver 모델 {#perceiver-model}

Perceiver 모델은 (공유 가중치와 스킵 연결을 통해) 크로스 어텐션과
트랜스포머 모듈 `num_iterations`번 반복하여,
잠재 배열이 필요에 따라 입력 이미지에서 정보를 반복적으로 추출할 수 있도록 합니다.

```python
class Perceiver(keras.Model):
    def __init__(
        self,
        patch_size,
        data_dim,
        latent_dim,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        ffn_units,
        dropout_rate,
        num_iterations,
        classifier_units,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.num_iterations = num_iterations
        self.classifier_units = classifier_units

    def build(self, input_shape):
        # 잠재 배열 생성.
        self.latent_array = self.add_weight(
            shape=(self.latent_dim, self.projection_dim),
            initializer="random_normal",
            trainable=True,
        )

        # module.war 패치 생성
        self.patch_encoder = PatchEncoder(self.data_dim, self.projection_dim)

        # 크로스 어텐션 모듈을 생성.
        self.cross_attention = create_cross_attention_module(
            self.latent_dim,
            self.data_dim,
            self.projection_dim,
            self.ffn_units,
            self.dropout_rate,
        )

        # 트랜스포머 모듈 생성.
        self.transformer = create_transformer_module(
            self.latent_dim,
            self.projection_dim,
            self.num_heads,
            self.num_transformer_blocks,
            self.ffn_units,
            self.dropout_rate,
        )

        # 글로벌 평균 풀링 레이어 생성.
        self.global_average_pooling = layers.GlobalAveragePooling1D()

        # 분류 헤드 생성.
        self.classification_head = create_ffn(
            hidden_units=self.classifier_units, dropout_rate=self.dropout_rate
        )

        super().build(input_shape)

    def call(self, inputs):
        # 데이터 보강.
        augmented = data_augmentation(inputs)
        # 패치 생성.
        patches = self.patcher(augmented)
        # 패치 인코딩.
        encoded_patches = self.patch_encoder(patches)
        # 크로스 어텐션 입력 준비.
        cross_attention_inputs = {
            "latent_array": ops.expand_dims(self.latent_array, 0),
            "data_array": encoded_patches,
        }
        # 크로스 어텐션과 트랜스포머 모듈을 반복적으로 적용합니다.
        for _ in range(self.num_iterations):
            # 잠재 배열에서 데이터 배열로 크로스 어텐션을 적용합니다.
            latent_array = self.cross_attention(cross_attention_inputs)
            # 잠재 배열에 셀프 어텐션 트랜스포머를 적용합니다.
            latent_array = self.transformer(latent_array)
            # 다음 반복의 잠재 배열을 설정합니다.
            cross_attention_inputs["latent_array"] = latent_array

        # 글로벌 평균 풀링을 적용하여, [batch_size, projection_dim] repesentation 텐서를 생성합니다.
        representation = self.global_average_pooling(latent_array)
        # logits을 생성.
        logits = self.classification_head(representation)
        return logits
```

## 모드 컴파일, 트레이닝 및 평가하기 {#compile-train-and-evaluate-the-mode}

```python
def run_experiment(model):
    # LAMB 옵티마이저 대신 가중치 감쇠가 있는 ADAM을 생성합니다. (LAMB는 아직 지원되지 않습니다.)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # 모델 컴파일.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )

    # 학습률 스케줄러 콜백을 생성합니다.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )

    # 조기 종료 콜백을 만듭니다.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    # 모델 Fit.
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # 학습 곡선을 그리기 위해 히스토리를 반환합니다.
    return history
```

V100 GPU의 현재 설정으로 perceiver 모델을 트레이닝하는데 약 200초가 걸립니다.

```python
perceiver_classifier = Perceiver(
    patch_size,
    num_patches,
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    ffn_units,
    dropout_rate,
    num_iterations,
    classifier_units,
)


history = run_experiment(perceiver_classifier)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Test accuracy: 0.91%
Test top 5 accuracy: 5.2%
```

{{% /details %}}

40 에포크가 지난 후, Perceiver 모델은 테스트 데이터에 대해
약 53%의 정확도와 81%의 top-5 정확도를 달성했습니다.

[Perceiver](https://arxiv.org/abs/2103.03206) 논문의 절제 연구에서 언급했듯이,
잠재 배열의 크기를 늘리고, 잠재 배열과 데이터 배열 요소의 (프로젝션) 차원을 늘리고,
트랜스포머 모듈의 블록 수를 늘리고,
크로스 어텐션과 잠재 트랜스포머 모듈을 적용하는 반복 횟수를 늘리면,
더 나은 결과를 얻을 수 있습니다.
입력 이미지의 크기를 늘리고 다른 패치 크기를 사용할 수도 있습니다.

Perceiver는 모델 크기를 늘리면 이점이 있습니다.
그러나, 모델이 커질수록 효율적으로 트레이닝하려면 더 큰 가속기가 필요합니다.
이것이 바로 Perceiver 논문에서 실험을 실행하기 위해, 32개의 TPU 코어를 사용한 이유입니다.
