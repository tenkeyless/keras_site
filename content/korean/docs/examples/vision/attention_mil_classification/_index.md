---
title: 어텐션 기반 심층 다중 인스턴스 학습(MIL)을 사용한 분류
linkTitle: MIL을 사용한 분류
toc: true
weight: 5
type: docs
math: true
---

{{< keras/original checkedAt="2024-11-19" >}}

**{{< t f_author >}}** [Mohamad Jaber](https://www.linkedin.com/in/mohamadjaber1/)  
**{{< t f_date_created >}}** 2021/08/16  
**{{< t f_last_modified >}}** 2021/11/25  
**{{< t f_description >}}** MIL 접근 방식을 사용하여 인스턴스 가방을 분류하고 개별 인스턴스 점수를 얻습니다.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/attention_mil_classification.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/attention_mil_classification.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

### 다중 인스턴스 학습(MIL, Multiple Instance Learning)이란 무엇인가요? {#what-is-multiple-instance-learning-mil}

일반적으로, 지도 학습 알고리즘을 사용하면, 학습자는 인스턴스 집합에 대한 레이블을 받습니다.
MIL의 경우, 학습자는 인스턴스 집합을 포함하는 가방 집합에 대한 레이블을 받습니다.
가방에 양성 인스턴스가 하나 이상 포함되어 있으면 양성,
하나도 포함되어 있지 않으면 음성으로 레이블이 지정됩니다.

### 동기 {#motivation}

이미지 분류 작업에서는 각 이미지가 클래스 레이블을 명확하게 나타낸다고 가정하는 경우가 많습니다.
의료 영상(예: 컴퓨터 병리학 등)에서는 _전체 이미지_ 가 단일 클래스 레이블(암/비암)로 표현되거나
관심 영역이 주어질 수 있습니다.
그러나, 이미지의 어떤 패턴이 실제로 해당 클래스에 속하게 만드는지 알고 싶을 것입니다.
이러한 맥락에서, 이미지를 분할하고 서브 이미지가 인스턴스 가방을 형성하게 할 것입니다.

따라서, 목표는 다음과 같습니다:

1.  인스턴스 가방에 대한 클래스 레이블을 예측하는 모델을 학습합니다.
2.  가방 내에서 어떤 인스턴스가 위치 클래스 레이블 예측을 일으켰는지 알아냅니다.

### 구현 {#implementation}

다음 단계에서는 모델 작동 방식을 설명합니다:

1.  특성 추출기 레이어가 특성 임베딩을 추출합니다.
2.  임베딩은 MIL 어텐션 레이어에 공급되어 어텐션 점수를 얻습니다. 이 레이어는 순열 불변형(permutation-invariant)으로 설계됩니다.
3.  입력 특성과 해당 어텐션 점수를 함께 곱합니다.
4.  결과 출력은 분류를 위해 소프트맥스 함수로 전달됩니다.

### 참조 {#references}

- [어텐션 기반 심층 다중 인스턴스 학습](https://arxiv.org/abs/1802.04712).
- 어텐션 연산자 코드 구현 중 일부는 [https://github.com/utayao/Atten_Deep_MIL](https://github.com/utayao/Atten_Deep_MIL)에서 영감을 얻었습니다.
- TensorFlow에서 제공하는 불균형 데이터 [튜토리얼](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)

## 셋업 {#setup}

```python
import numpy as np
import keras
from keras import layers
from keras import ops
from tqdm import tqdm
from matplotlib import pyplot as plt

plt.style.use("ggplot")
```

## 데이터 세트 생성 {#create-dataset}

가방 세트를 생성하고 내용물에 따라 라벨을 할당합니다.
가방에 하나 이상의 양성 인스턴스가 있으면 해당 가방은 양성 가방으로 간주됩니다.
양성 인스턴스가 하나도 포함되어 있지 않으면 해당 가방은 음성 가방으로 간주됩니다.

### 구성 매개변수 {#configuration-parameters}

- `POSITIVE_CLASS`: 포지티브 가방에 보관할 원하는 클래스입니다.
- `BAG_COUNT`: 트레이닝 가방의 개수입니다.
- `VAL_BAG_COUNT`: 검증 가방의 개수입니다.
- `BAG_SIZE`: 가방의 인스턴스 수입니다.
- `PLOT_SIZE`: 플롯할 가방의 수입니다.
- `ENSEMBLE_AVG_COUNT`: 함께 생성하고 평균을 낼 모델의 수입니다. (선택 사항: 종종 더 나은 성능을 제공합니다. 단일 모델의 경우 1로 설정).

```python
POSITIVE_CLASS = 1
BAG_COUNT = 1000
VAL_BAG_COUNT = 300
BAG_SIZE = 3
PLOT_SIZE = 3
ENSEMBLE_AVG_COUNT = 1
```

### 가방 준비하기 {#prepare-bags}

어텐션 연산자는 순열 불변(permutation-invariant) 연산자이므로,
클래스 레이블이 양성인 인스턴스는 양성 가방에 있는 인스턴스 중에서 무작위로 배치됩니다.

```python
def create_bags(input_data, input_labels, positive_class, bag_count, instance_count):
    # 가방 셋업.
    bags = []
    bag_labels = []

    # 입력 데이터 정규화.
    input_data = np.divide(input_data, 255.0)

    # 양성 샘플을 계산.
    count = 0

    for _ in range(bag_count):
        # 샘플의 고정된 크기 무작위 서브 집합을 선택.
        index = np.random.choice(input_data.shape[0], instance_count, replace=False)
        instances_data = input_data[index]
        instances_labels = input_labels[index]

        # 기본적으로, 모든 가방은 0으로 레이블이 지정.
        bag_label = 0

        # 가방에 적어도 하나의 양성 클래스가 있는지 확인.
        if positive_class in instances_labels:
            # 양성 가방은 1로 표시.
            bag_label = 1
            count += 1

        bags.append(instances_data)
        bag_labels.append(np.array([bag_label]))

    print(f"Positive bags: {count}")
    print(f"Negative bags: {bag_count - count}")

    return (list(np.swapaxes(bags, 0, 1)), np.array(bag_labels))


# MNIST 데이터 세트를 로드.
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

# 트레이닝 데이터를 생성.
train_data, train_labels = create_bags(
    x_train, y_train, POSITIVE_CLASS, BAG_COUNT, BAG_SIZE
)

# 검증 데이터를 생성.
val_data, val_labels = create_bags(
    x_val, y_val, POSITIVE_CLASS, VAL_BAG_COUNT, BAG_SIZE
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Positive bags: 283
Negative bags: 717
Positive bags: 104
Negative bags: 196
```

{{% /details %}}

## 모델 만들기 {#create-the-model}

이제 어텐션 레이어를 빌드하고, 몇 가지 유틸리티를 준비한 다음, 전체 모델을 빌드하고 트레이닝하겠습니다.

### 어텐션 연산자 구현하기 {#attention-operator-implementation}

이 레이어의 출력 크기는 단일 가방의 크기에 따라 결정됩니다.

어텐션 메커니즘은 가방에 있는 인스턴스의 가중 평균을 사용하며,
가중치의 합은 1(가방 크기의 불변성)이어야 합니다.

가중치 행렬(매개변수)은 **w**와 **v**입니다.
양수 및 음수 값을 포함하기 위해, 쌍곡선 탄젠트(hyperbolic tangent) 요소별 비선형성이 활용됩니다.

복잡한 관계를 처리하기 위해 **게이트 어텐션 메커니즘**을 사용할 수 있습니다.
또다른 가중치 행렬인, **u**가 계산에 추가됩니다.
쌍곡선 탄젠트 비선형성을 통해 $x \in [-1, 1]$에 대한 거의 선형적인 동작을 극복하기 위해
시그모이드 비선형성이 사용됩니다.

```python
class MILAttentionLayer(layers.Layer):
    """어텐션 기반 딥 MIL 레이어 구현.

    Args:
      weight_params_dim: 양의 정수. 가중치 행렬의 차원입니다.
      kernel_initializer: `kernel` 행렬의 이니셜라이저.
      kernel_regularizer: `kernel` 행렬에 적용될 정규화 함수.
      use_gated: Boolean. 게이트 메커니즘을 사용할지 여부입니다.

    Returns:
      BAG_SIZE 길이의 2D 텐서 리스트.
      텐서는 `(batch_size, 1)` 모양의 소프트맥스 이후의 어텐션 점수입니다.
    """

    def __init__(
        self,
        weight_params_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):
        # 입력 모양.
        # 다음 모양의 2D 텐서 리스트: (batch_size, input_dim).
        input_dim = input_shape[0][1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):
        # 입력 수에서 변수를 할당합니다.
        instances = [self.compute_attention_scores(instance) for instance in inputs]

        # 인스턴스를 단일 텐서로 스택합니다.
        instances = ops.stack(instances)

        # 출력 합이 1이 되도록 인스턴스에 걸쳐 소프트맥스를 적용합니다.
        alpha = ops.softmax(instances, axis=0)

        # 분할하여 입력으로 사용한 것과 동일한 텐서 배열을 다시 생성합니다.
        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_attention_scores(self, instance):
        # "게이트 메커니즘"이 사용될 경우를 위해 예약합니다.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = ops.tanh(ops.tensordot(instance, self.v_weight_params, axes=1))

        # 비선형 관계를 효율적으로 학습하기 위함.
        if self.use_gated:
            instance = instance * ops.sigmoid(
                ops.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return ops.tensordot(instance, self.w_weight_params, axes=1)
```

## 시각화 도구 {#visualizer-tool}

클래스에 대한 가방 수(`PLOT_SIZE`로 제공됨)를 플롯합니다.

또한, 활성화하면, 각 가방에 대한 관련 인스턴스 점수와 함께
클래스 레이블 예측(모델 트레이닝된 후)을 볼 수 있습니다.

```python
def plot(data, labels, bag_class, predictions=None, attention_weights=None):
    """가방과 어텐션 가중치를 플로팅하는 유틸리티.

    Args:
      data: 인스턴스 가방이 포함된 입력 데이터.
      labels: 입력 데이터의 연관된 가방 레이블.
      bag_class: 원하는 가방 클래스의 문자열 이름.
        옵션은 다음과 같습니다: "positive" 또는 "negative".
      predictions: 클래스 레이블 모델 예측.
        아무것도 지정하지 않으면, ground truth 레이블이 사용됩니다.
      attention_weights: 입력 데이터 내 각 인스턴스에 대한 어텐션 가중치.
        아무것도 지정하지 않으면, 값이 표시되지 않습니다.
    """
    return  ## TODO
    labels = np.array(labels).reshape(-1)

    if bag_class == "positive":
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

        else:
            labels = np.where(labels == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

    elif bag_class == "negative":
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
        else:
            labels = np.where(labels == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

    else:
        print(f"There is no class {bag_class}")
        return

    print(f"The bag class label is {bag_class}")
    for i in range(PLOT_SIZE):
        figure = plt.figure(figsize=(8, 8))
        print(f"Bag number: {labels[i]}")
        for j in range(BAG_SIZE):
            image = bags[j][i]
            figure.add_subplot(1, BAG_SIZE, j + 1)
            plt.grid(False)
            if attention_weights is not None:
                plt.title(np.around(attention_weights[labels[i]][j], 2))
            plt.imshow(image)
        plt.show()


# 클래스별 검증 데이터 가방을 플롯합니다.
plot(val_data, val_labels, "positive")
plot(val_data, val_labels, "negative")
```

## 모델 만들기 {#create-model}

먼저 인스턴스당 몇 개의 임베딩을 생성하고, 어텐션 연산자를 호출한 다음,
소프트맥스 함수를 사용하여 클래스 확률을 출력합니다.

```python
def create_model(instance_shape):
    # 입력에서 특성을 추출.
    inputs, embeddings = [], []
    shared_dense_layer_1 = layers.Dense(128, activation="relu")
    shared_dense_layer_2 = layers.Dense(64, activation="relu")
    for _ in range(BAG_SIZE):
        inp = layers.Input(instance_shape)
        flatten = layers.Flatten()(inp)
        dense_1 = shared_dense_layer_1(flatten)
        dense_2 = shared_dense_layer_2(dense_1)
        inputs.append(inp)
        embeddings.append(dense_2)

    # 어텐션 레이어를 호출.
    alpha = MILAttentionLayer(
        weight_params_dim=256,
        kernel_regularizer=keras.regularizers.L2(0.01),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # 입력 레이어에 어텐션 가중치를 곱함.
    multiply_layers = [
        layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    ]

    # 레이어 Concatenate.
    concat = layers.concatenate(multiply_layers, axis=1)

    # 분류 출력 노드.
    output = layers.Dense(2, activation="softmax")(concat)

    return keras.Model(inputs, output)
```

## 클래스 가중치 {#class-weights}

이러한 종류의 문제는 단순히 불균형한 데이터 분류 문제로 바뀔 수 있으므로, 클래스 가중치를 고려해야 합니다.

가방이 1000개 있다고 가정해 보겠습니다.
가방의 90%에는 양성 라벨이 하나도 없고, 10%에만 양성 라벨이 있는 경우가 종종 있을 수 있습니다.
이러한 데이터를 **불균형 데이터(Imbalanced data)**라고 할 수 있습니다.

클래스 가중치를 사용하면, 모델은 희귀 클래스에 더 높은 가중치를 부여하는 경향이 있습니다.

```python
def compute_class_weights(labels):
    # 양성 및 음성 가방 개수를 계산.
    negative_count = len(np.where(labels == 0)[0])
    positive_count = len(np.where(labels == 1)[0])
    total_count = negative_count + positive_count

    # 클래스 가중치 딕셔너리 빌드.
    return {
        0: (1 / negative_count) * (total_count / 2),
        1: (1 / positive_count) * (total_count / 2),
    }
```

## 모델 빌드 및 트레이닝 {#build-and-train-model}

이 섹션에서는 모델을 빌드하고 트레이닝합니다.

```python
def train(train_data, train_labels, val_data, val_labels, model):
    # 모델 트레이닝.
    # 콜백을 준비합니다.
    # 최적의 가중치를 저장할 경로를 설정합니다.

    # 래퍼에서 파일 이름을 가져옵니다.
    file_path = "/tmp/best_model.weights.h5"

    # 모델 체크포인트 콜백을 초기화합니다.
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose=0,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    # 조기 중지(early stopping) 콜백을 초기화합니다.
    # 검증 데이터 전반에 걸쳐 모델 성능을 모니터링하고,
    # 일반화 오류가 더 이상 감소하지 않으면 트레이닝을 중지합니다.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    # 모델 컴파일.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 모델 Fit.
    model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=20,
        class_weight=compute_class_weights(train_labels),
        batch_size=1,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0,
    )

    # 최적의 가중치를 로드.
    model.load_weights(file_path)

    return model


# 모델(들) 빌드.
instance_shape = train_data[0][0].shape
models = [create_model(instance_shape) for _ in range(ENSEMBLE_AVG_COUNT)]

# 단일 모델 아키텍처를 표시.
print(models[0].summary())

# 모델(들) 트레이닝.
trained_models = [
    train(train_data, train_labels, val_data, val_labels, model)
    for model in tqdm(models)
]
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃ Param # ┃ Connected to         ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ input_layer         │ (None, 28, 28)    │       0 │ -                    │
│ (InputLayer)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ input_layer_1       │ (None, 28, 28)    │       0 │ -                    │
│ (InputLayer)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ input_layer_2       │ (None, 28, 28)    │       0 │ -                    │
│ (InputLayer)        │                   │         │                      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ flatten (Flatten)   │ (None, 784)       │       0 │ input_layer[0][0]    │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ flatten_1 (Flatten) │ (None, 784)       │       0 │ input_layer_1[0][0]  │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ flatten_2 (Flatten) │ (None, 784)       │       0 │ input_layer_2[0][0]  │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dense (Dense)       │ (None, 128)       │ 100,480 │ flatten[0][0],       │
│                     │                   │         │ flatten_1[0][0],     │
│                     │                   │         │ flatten_2[0][0]      │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dense_1 (Dense)     │ (None, 64)        │   8,256 │ dense[0][0],         │
│                     │                   │         │ dense[1][0],         │
│                     │                   │         │ dense[2][0]          │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ alpha               │ [(None, 1),       │  33,024 │ dense_1[0][0],       │
│ (MILAttentionLayer) │ (None, 1), (None, │         │ dense_1[1][0],       │
│                     │ 1)]               │         │ dense_1[2][0]        │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ multiply (Multiply) │ (None, 64)        │       0 │ alpha[0][0],         │
│                     │                   │         │ dense_1[0][0]        │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ multiply_1          │ (None, 64)        │       0 │ alpha[0][1],         │
│ (Multiply)          │                   │         │ dense_1[1][0]        │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ multiply_2          │ (None, 64)        │       0 │ alpha[0][2],         │
│ (Multiply)          │                   │         │ dense_1[2][0]        │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ concatenate         │ (None, 192)       │       0 │ multiply[0][0],      │
│ (Concatenate)       │                   │         │ multiply_1[0][0],    │
│                     │                   │         │ multiply_2[0][0]     │
├─────────────────────┼───────────────────┼─────────┼──────────────────────┤
│ dense_2 (Dense)     │ (None, 2)         │     386 │ concatenate[0][0]    │
└─────────────────────┴───────────────────┴─────────┴──────────────────────┘
 Total params: 142,146 (555.26 KB)
 Trainable params: 142,146 (555.26 KB)
 Non-trainable params: 0 (0.00 B)
```

```plain
None

100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:36<00:00, 36.67s/it]
```

{{% /details %}}

## 모델 평가 {#model-evaluation}

이제 모델을 평가할 준비가 되었습니다.
각 모델에 대해 어텐션 레이어에서 가중치를 얻기 위해 연결된 중간 모델도 생성합니다.

각 `ENSEMBLE_AVG_COUNT` 모델에 대한 예측을 계산하고, 최종 예측을 위해 함께 평균을 냅니다.

```python
def predict(data, labels, trained_models):
    # 모델별로 정보를 수집.
    models_predictions = []
    models_attention_weights = []
    models_losses = []
    models_accuracies = []

    for model in trained_models:
        # 데이터에 대한 출력 클래스를 예측.
        predictions = model.predict(data)
        models_predictions.append(predictions)

        # 중간 모델을 생성하여 MIL 어텐션 레이어 가중치 얻기.
        intermediate_model = keras.Model(model.input, model.get_layer("alpha").output)

        # MIL 어텐션 레이어 가중치를 예측.
        intermediate_predictions = intermediate_model.predict(data)

        attention_weights = np.squeeze(np.swapaxes(intermediate_predictions, 1, 0))
        models_attention_weights.append(attention_weights)

        loss, accuracy = model.evaluate(data, labels, verbose=0)
        models_losses.append(loss)
        models_accuracies.append(accuracy)

    print(
        f"The average loss and accuracy are {np.sum(models_losses, axis=0) / ENSEMBLE_AVG_COUNT:.2f}"
        f" and {100 * np.sum(models_accuracies, axis=0) / ENSEMBLE_AVG_COUNT:.2f} % resp."
    )

    return (
        np.sum(models_predictions, axis=0) / ENSEMBLE_AVG_COUNT,
        np.sum(models_attention_weights, axis=0) / ENSEMBLE_AVG_COUNT,
    )


# 검증 데이터에 대해 클래스 및 어텐션 점수를 평가하고 예측.
class_predictions, attention_params = predict(val_data, val_labels, trained_models)

# 검증 데이터의 몇 가지 결과를 플로팅.
plot(
    val_data,
    val_labels,
    "positive",
    predictions=class_predictions,
    attention_weights=attention_params,
)
plot(
    val_data,
    val_labels,
    "negative",
    predictions=class_predictions,
    attention_weights=attention_params,
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 10/10 ━━━━━━━━━━━━━━━━━━━━ 1s 53ms/step
 10/10 ━━━━━━━━━━━━━━━━━━━━ 1s 39ms/step
The average loss and accuracy are 0.03 and 99.00 % resp.
```

{{% /details %}}

## 결론 {#conclusion}

위의 플롯에서, 가중치가 항상 1로 합산되는 것을 볼 수 있습니다.
양성으로 예측된 가방에서는, 양성으로 라벨링된 인스턴스가
나머지 가방보다 훨씬 더 높은 어텐션 점수를 갖게 됩니다.
그러나, 음성으로 예측된 가방에는 두 가지 경우가 있습니다:

- 모든 인스턴스의 점수가 거의 비슷합니다.
- 한 인스턴스의 점수가 상대적으로 더 높지만, 양성 인스턴스만큼 높지는 않습니다.
  이는 이 인스턴스의 특성 공간이 양성 인스턴스의 특성 공간에 가깝기 때문입니다.

## 비고 {#remarks}

- 모델이 과적합하면, 모든 가방에 가중치가 동일하게 분포됩니다. 따라서, 정규화 기법이 필요합니다.
- 본 논문에서, 가방 크기는 가방마다 다를 수 있습니다. 편의를 위해, 여기서는 가방 크기를 고정했습니다.
- 단일 모델의 무작위 초기 가중치에 의존하지 않으려면, 평균 앙상블 방법을 고려해야 합니다.
