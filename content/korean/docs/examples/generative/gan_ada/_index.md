---
title: 적응형 판별자 보강을 통한 데이터 효율적 GAN
linkTitle: 데이터 효율적 GAN
toc: true
weight: 12
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** [András Béres](https://www.linkedin.com/in/andras-beres-789190210)  
**{{< t f_date_created >}}** 2021/10/28  
**{{< t f_last_modified >}}** 2021/10/28  
**{{< t f_description >}}** Caltech Birds 데이터셋을 사용하여 제한된 데이터로 이미지 생성하기

{{< keras/version v=2 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/gan_ada.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/gan_ada.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

### GANs {#gans}

[생성적 적대 신경망(GANs)](https://arxiv.org/abs/1406.2661)은 이미지 생성을 위해 자주 사용되는,
생성적 딥러닝 모델의 인기 있는 클래스입니다.
GAN은 판별자와 생성자라고 불리는 두 개의 대립하는 신경망으로 구성됩니다.
판별자의 작업은 실제 이미지와 생성된(가짜) 이미지를 구분하는 것이며,
생성자 네트워크는 점점 더 현실적인 이미지를 생성하여 판별자를 속이려고 합니다.
그러나 생성자를 속이는 것이 너무 쉽거나 너무 어려운 경우,
생성자에게 유용한 학습 신호를 제공하지 못할 수 있기 때문에,
GAN의 트레이닝은 일반적으로 어려운 작업으로 간주됩니다.

### GAN을 위한 데이터 보강 {#data-augmentation-for-gans}

데이터 보강은 딥러닝에서 널리 사용되는 기법으로,
입력 데이터에 의미를 보존하는 변환을 무작위로 적용하여 여러 개의 현실적인 버전을 생성함으로써,
사용 가능한 트레이닝 데이터의 양을 효과적으로 늘리는 과정입니다.
가장 간단한 예로는 이미지를 좌우 반전시키는 것으로,
이는 그 내용물을 유지하면서 두 번째 고유한 트레이닝 샘플을 생성합니다.
데이터 보강은 일반적으로 지도 학습에서 과적합을 방지하고 일반화 성능을 향상시키기 위해 사용됩니다.

[StyleGAN2-ADA](https://arxiv.org/abs/2006.06676)의 저자들은,
특히 트레이닝 데이터의 양이 적을 때,
판별자의 과적합이 GAN에서 문제가 될 수 있음을 보여주었습니다.
그들은 이러한 문제를 완화하기 위해 Adaptive Discriminator 보강을 제안했습니다.

그러나 GAN에 데이터 보강을 적용하는 것은 간단하지 않습니다.
생성자는 판별자의 그래디언트를 사용하여 업데이트되기 때문에,
생성된 이미지에 보강이 적용될 경우,
보강 파이프라인은 미분 가능해야 하고 계산 효율성을 위해 GPU와도 호환되어야 합니다.
다행히 [Keras 이미지 보강 레이어]({{< relref "/docs/api/layers/preprocessing_layers/image_augmentation" >}})는
이러한 두 가지 요구 사항을 모두 충족하므로, 이 작업에 매우 적합합니다.

### 가역적(Invertible) 데이터 보강 {#invertible-data-augmentation}

생성 모델에서 데이터 보강을 사용할 때 발생할 수 있는 어려움 중 하나는
["누출되는(leaky) 보강"(섹션 2.2)](https://arxiv.org/abs/2006.06676) 문제입니다.
이는 모델이 이미 보강된 이미지를 생성하는 경우를 말합니다.
이는 모델이 보강과 기본 데이터 분포를 분리할 수 없었다는 것을 의미하며,
가역적이지 않은(non-invertible) 데이터 변환을 사용할 때 발생할 수 있습니다.
예를 들어, 0도, 90도, 180도, 270도의 회전이 동일한 확률로 수행될 경우,
이미지의 원래 방향을 추론하는 것이 불가능해지며, 이 정보는 손실됩니다.

데이터 보강을 가역적으로 만드는 간단한 방법은 보강을 일정한 확률로만 적용하는 것입니다.
이렇게 하면 원본 버전의 이미지가 더 많이 나타나고, 데이터 분포를 추론할 수 있습니다.
이 확률을 적절히 선택함으로써 보강이 누출되지 않도록 하면서, 판별자를 효과적으로 정규화할 수 있습니다.

## 셋업 {#setup}

```python
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
```

## 하이퍼파라미터 {#hyperparameterers}

```python
# 데이터
num_epochs = 10  # 좋은 결과를 위해 400 에포크 동안 트레이닝
image_size = 64
# Kernel Inception Distance 측정 해상도, 관련 섹션 참조
kid_image_size = 75
padding = 0.25
dataset_name = "caltech_birds2011"

# 적응형 판별자 보강
max_translation = 0.125
max_rotation = 0.125
max_zoom = 0.25
target_accuracy = 0.85
integration_steps = 1000

# 아키텍처
noise_size = 64
depth = 4
width = 128
leaky_relu_slope = 0.2
dropout_rate = 0.4

# 최적화
batch_size = 128
learning_rate = 2e-4
beta_1 = 0.5  # 기본값 0.9를 사용하지 않는 것이 중요합니다
ema = 0.99
```

## 데이터 파이프라인 {#data-pipeline}

이 예제에서는 [Caltech Birds (2011)](https://www.tensorflow.org/datasets/catalog/caltech_birds2011) 데이터셋을 사용하여 새의 이미지를 생성할 것입니다.
이 데이터셋은 트레이닝을 위해 6000개 미만의 이미지를 포함하고 있는 다양한 자연 데이터셋입니다.
이처럼 데이터 양이 적을 때는, 가능한 한 높은 데이터 품질을 유지하기 위해 특별한 주의를 기울여야 합니다.
이 예제에서는, 제공된 새의 바운딩 박스를 사용하여 가능한 경우, 종횡비를 유지하면서 정사각형으로 자릅니다.

```python
def round_to_int(float_value):
    return tf.cast(tf.math.round(float_value), dtype=tf.int32)


def preprocess_image(data):
    # 바운딩 박스 좌표 비정규화
    height = tf.cast(tf.shape(data["image"])[0], dtype=tf.float32)
    width = tf.cast(tf.shape(data["image"])[1], dtype=tf.float32)
    bounding_box = data["bbox"] * tf.stack([height, width, height, width])

    # 중심과 긴 변의 길이를 계산하고 패딩을 추가
    target_center_y = 0.5 * (bounding_box[0] + bounding_box[2])
    target_center_x = 0.5 * (bounding_box[1] + bounding_box[3])
    target_size = tf.maximum(
        (1.0 + padding) * (bounding_box[2] - bounding_box[0]),
        (1.0 + padding) * (bounding_box[3] - bounding_box[1]),
    )

    # 크롭 크기를 이미지에 맞게 조정
    target_height = tf.reduce_min(
        [target_size, 2.0 * target_center_y, 2.0 * (height - target_center_y)]
    )
    target_width = tf.reduce_min(
        [target_size, 2.0 * target_center_x, 2.0 * (width - target_center_x)]
    )

    # 이미지 크롭
    image = tf.image.crop_to_bounding_box(
        data["image"],
        offset_height=round_to_int(target_center_y - 0.5 * target_height),
        offset_width=round_to_int(target_center_x - 0.5 * target_width),
        target_height=round_to_int(target_height),
        target_width=round_to_int(target_width),
    )

    # 크기 조정 및 클리핑
    # 이미지 다운샘플링의 경우, 영역 보간법이 선호됩니다.
    image = tf.image.resize(
        image, size=[image_size, image_size], method=tf.image.ResizeMethod.AREA
    )
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_dataset(split):
    # 검증 데이터셋도 셔플합니다. 데이터 순서가 KID 계산에 중요하기 때문입니다.
    return (
        tfds.load(dataset_name, split=split, shuffle_files=True)
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


train_dataset = prepare_dataset("train")
val_dataset = prepare_dataset("test")
```

전처리 후의 트레이닝 이미지는 다음과 같습니다:

![birds dataset](/images/examples/generative/gan_ada/Ru5HgBM.png)

## Kernel inception distance {#kernel-inception-distance}

[Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401)는
이미지 생성 품질을 측정하기 위해 널리 사용되는
[Frechet Inception Distance (FID)](https://arxiv.org/abs/1706.08500) 메트릭의 대체로 제안되었습니다.
두 메트릭 모두 [ImageNet](https://www.tensorflow.org/datasets/catalog/imagenet2012)에 대해
사전 트레이닝된 [InceptionV3]({{< relref "/docs/api/applications/inceptionv3" >}}) 네트워크의 표현 공간에서,
생성된 데이터와 트레이닝 데이터 분포 간의 차이를 측정합니다.

논문에 따르면, FID는 편향되지 않은 추정치가 없기 때문에, 적은 수의 이미지로 측정할 때 기대값이 더 높아집니다.
KID는 측정된 샘플 수에 관계없이 기대값이 변하지 않기 때문에 작은 데이터셋에 더 적합합니다.
제 경험으로는 KID가 계산적으로 더 가볍고, 수치적으로 더 안정적이며, 배치 단위로 추정할 수 있기 때문에 구현이 더 간단합니다.

이 예제에서는, Inception 네트워크의 최소 가능한 해상도(75x75 대신 299x299)로 이미지를 평가하고,
계산 효율성을 위해 검증 세트에서만 메트릭을 측정합니다.

```python
class KID(keras.metrics.Metric):
    def __init__(self, name="kid", **kwargs):
        super().__init__(name=name, **kwargs)

        # KID는 배치별로 추정되며 배치 간 평균이 산출됩니다.
        self.kid_tracker = keras.metrics.Mean()

        # 사전 트레이닝된 InceptionV3가 분류 레이어 없이 사용됩니다.
        # 픽셀 값을 0-255 범위로 변환한 다음,
        # 사전 트레이닝 시와 동일한 전처리를 사용합니다.
        self.encoder = keras.Sequential(
            [
                layers.InputLayer(input_shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # 두 특징 집합을 사용하여 다항식 커널을 계산합니다.
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)
        # 평균 커널 값을 사용하여 최대 평균 차이의 제곱(squared maximum mean discrepancy)을 추정합니다.
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # 평균 KID 추정치를 업데이트합니다.
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()
```

## 적응형 판별자 보강 {#adaptive-discriminator-augmentation}

[StyleGAN2-ADA](https://arxiv.org/abs/2006.06676)의 저자들은
트레이닝 중에 보강 확률을 적응적으로 변경할 것을 제안합니다.
논문에서는 다르게 설명되어 있지만,
그들은 보강 확률에 대해 [적분 제어](https://en.wikipedia.org/wiki/PID_controller#Integral)를 사용하여,
판별자의 실제 이미지에 대한 정확도가 목표 값에 가깝도록 유지합니다.
주의할 점은 그들이 제어하는 변수는,
실제로 판별자 로짓의 평균 부호(r_t로 논문에서 표시됨)이며, 이는 2 \* accuracy - 1에 해당합니다.

이 메서드는 두 개의 하이퍼파라미터를 필요로 합니다:

1. `target_accuracy`: 판별자의 실제 이미지에 대한 정확도의 목표 값입니다.
   이 값을 80-90% 범위에서 선택할 것을 추천합니다.
2. [`integration_steps`](https://en.wikipedia.org/wiki/PID_controller#Mathematical_form):
   정확도 에러가 100%일 때 보강 확률 증가가 100%가 되도록 하는 업데이트 스텝의 수입니다.
   직관적으로, 이는 보강 확률이 얼마나 천천히 변경되는지를 정의합니다.
   보강 강도가 천천히 조정되도록, 이 값을 비교적 큰 값(이 경우 1000)으로 설정할 것을 추천합니다.

이 절차의 주요 동기는 목표 정확도의 최적 값이 다양한 데이터셋 크기에서 유사하다는 것입니다.
(논문의 [그림 4와 5](https://arxiv.org/abs/2006.06676)를 참조하십시오)
따라서 재튜닝이 필요하지 않으며, 이 프로세스는 필요할 때 자동으로 더 강한 데이터 보강을 적용합니다.

```python
# "hard sigmoid", 로짓에서 이진 정확도 계산에 유용합니다.
def step(values):
    # 음수 값 -> 0.0, 양수 값 -> 1.0
    return 0.5 * (1.0 + tf.sign(values))


# 트레이닝 중에 동적으로 업데이트되는 확률로 이미지를 보강합니다.
class AdaptiveAugmenter(keras.Model):
    def __init__(self):
        super().__init__()

        # 이미지가 보강될 현재 확률을 저장합니다.
        self.probability = tf.Variable(0.0)
        # 논문에서 언급된 해당 보강 이름이 각 레이어 위에 표시되어 있습니다.
        # 저자들은 (그림 4 참조) 블리팅(blitting)과 기하학적 보강이
        # 적은 데이터 상황에서 가장 도움이 된다고 보여줍니다.
        self.augmenter = keras.Sequential(
            [
                layers.InputLayer(input_shape=(image_size, image_size, 3)),
                # blitting/x-flip:
                layers.RandomFlip("horizontal"),
                # blitting/integer 변환:
                layers.RandomTranslation(
                    height_factor=max_translation,
                    width_factor=max_translation,
                    interpolation="nearest",
                ),
                # geometric/rotation:
                layers.RandomRotation(factor=max_rotation),
                # geometric/isotropic 및 anisotropic 스케일링:
                layers.RandomZoom(
                    height_factor=(-max_zoom, 0.0), width_factor=(-max_zoom, 0.0)
                ),
            ],
            name="adaptive_augmenter",
        )

    def call(self, images, training):
        if training:
            augmented_images = self.augmenter(images, training)

            # 트레이닝 중에는 원본 이미지 또는 보강된 이미지가 self.probability에 따라 선택됩니다.
            augmentation_values = tf.random.uniform(
                shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
            )
            augmentation_bools = tf.math.less(augmentation_values, self.probability)

            images = tf.where(augmentation_bools, augmented_images, images)
        return images

    def update(self, real_logits):
        current_accuracy = tf.reduce_mean(step(real_logits))

        # 판별자의 실제 이미지에 대한 정확도를 기반으로 보강 확률이 업데이트됩니다.
        accuracy_error = current_accuracy - target_accuracy
        self.probability.assign(
            tf.clip_by_value(
                self.probability + accuracy_error / integration_steps, 0.0, 1.0
            )
        )
```

## 네트워크 아키텍쳐 {#network-architecture}

여기서는 두 네트워크의 아키텍처를 정의합니다:

- 생성자: 랜덤 벡터를 가능한 한 현실적인 이미지로 매핑합니다.
- 판별자: 이미지를 스칼라 점수로 매핑하며, 실제 이미지에는 높은 점수를, 생성된 이미지에는 낮은 점수를 부여합니다.

GAN은 네트워크 아키텍처에 민감한 경향이 있는데, 이 예제에서는 DCGAN 아키텍처를 구현했습니다.
DCGAN은 트레이닝 중에 비교적 안정적이며 구현이 간단하기 때문입니다.
네트워크 전체에서 필터 수를 일정하게 유지하고,
생성자의 마지막 레이어에서 `tanh` 대신 `sigmoid`를 사용하며,
랜덤 노말 초기화 대신 기본 초기화를 사용하는 등의 단순화를 적용했습니다.

좋은 관행으로서, 배치 정규화 레이어에서 학습 가능한 스케일 매개변수를 비활성화했습니다.
이는 한편으로는 뒤따르는 relu + 컨볼루션 레이어가 이를 중복되게 만들기 때문이고
([문서]({{< relref "/docs/api/layers/normalization_layers/batch_normalization" >}})에 언급된 것처럼),
다른 한편으로는 [스펙트럼 정규화(섹션 4.1)](https://arxiv.org/abs/1802.05957)를 사용할 때
이 매개변수를 비활성화해야 하기 때문입니다.
스펙트럼 정규화는 여기서는 사용하지 않지만, GAN에서는 일반적입니다.
또한, 배치 정규화가 뒤따라오기 때문에 완전 연결(fully connected) 레이어와 컨볼루션 레이어의 bias도 비활성화했습니다.

```python
# DCGAN 생성자
def get_generator():
    noise_input = keras.Input(shape=(noise_size,))
    x = layers.Dense(4 * 4 * width, use_bias=False)(noise_input)
    x = layers.BatchNormalization(scale=False)(x)
    x = layers.ReLU()(x)
    x = layers.Reshape(target_shape=(4, 4, width))(x)
    for _ in range(depth - 1):
        x = layers.Conv2DTranspose(
            width, kernel_size=4, strides=2, padding="same", use_bias=False,
        )(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.ReLU()(x)
    image_output = layers.Conv2DTranspose(
        3, kernel_size=4, strides=2, padding="same", activation="sigmoid",
    )(x)

    return keras.Model(noise_input, image_output, name="generator")


# DCGAN 판별자
def get_discriminator():
    image_input = keras.Input(shape=(image_size, image_size, 3))
    x = image_input
    for _ in range(depth):
        x = layers.Conv2D(
            width, kernel_size=4, strides=2, padding="same", use_bias=False,
        )(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.LeakyReLU(alpha=leaky_relu_slope)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    output_score = layers.Dense(1)(x)

    return keras.Model(image_input, output_score, name="discriminator")
```

## GAN 모델 {#gan-model}

```python
class GAN_ADA(keras.Model):
    def __init__(self):
        super().__init__()

        self.augmenter = AdaptiveAugmenter()
        self.generator = get_generator()
        self.ema_generator = keras.models.clone_model(self.generator)
        self.discriminator = get_discriminator()

        self.generator.summary()
        self.discriminator.summary()

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super().compile(**kwargs)

        # 두 네트워크에 대한 개별 옵티마이저
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.generator_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.discriminator_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.real_accuracy = keras.metrics.BinaryAccuracy(name="real_acc")
        self.generated_accuracy = keras.metrics.BinaryAccuracy(name="gen_acc")
        self.augmentation_probability_tracker = keras.metrics.Mean(name="aug_p")
        self.kid = KID()

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.real_accuracy,
            self.generated_accuracy,
            self.augmentation_probability_tracker,
            self.kid,
        ]

    def generate(self, batch_size, training):
        latent_samples = tf.random.normal(shape=(batch_size, noise_size))
        # 추론 중에는 ema_generator 사용
        if training:
            generated_images = self.generator(latent_samples, training)
        else:
            generated_images = self.ema_generator(latent_samples, training)
        return generated_images

    def adversarial_loss(self, real_logits, generated_logits):
        # 일반적으로 비포화 GAN 손실(non-saturating GAN loss)이라고 불립니다.
        real_labels = tf.ones(shape=(batch_size, 1))
        generated_labels = tf.zeros(shape=(batch_size, 1))

        # 생성자는 판별자가 실제라고 간주하는 이미지를 생성하려고 시도합니다.
        generator_loss = keras.losses.binary_crossentropy(
            real_labels, generated_logits, from_logits=True
        )

        # 판별자는 이미지가 실제인지 생성된 것인지를 결정하려고 합니다.
        discriminator_loss = keras.losses.binary_crossentropy(
            tf.concat([real_labels, generated_labels], axis=0),
            tf.concat([real_logits, generated_logits], axis=0),
            from_logits=True,
        )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)

    def train_step(self, real_images):
        real_images = self.augmenter(real_images, training=True)

        # 그래디언트를 두 번 계산할 것이므로 지속적인 그래디언트 테이프 사용
        with tf.GradientTape(persistent=True) as tape:
            generated_images = self.generate(batch_size, training=True)
            # 이미지 보강을 통해 그래디언트가 계산됩니다.
            generated_images = self.augmenter(generated_images, training=True)

            # 실제 이미지와 생성된 이미지에 대해 별도의 순전파가 수행되며,
            # 이는 배치 정규화가 별도로 적용됨을 의미합니다.
            real_logits = self.discriminator(real_images, training=True)
            generated_logits = self.discriminator(generated_images, training=True)

            generator_loss, discriminator_loss = self.adversarial_loss(
                real_logits, generated_logits
            )

        # 그래디언트를 계산하고 가중치를 업데이트
        generator_gradients = tape.gradient(
            generator_loss, self.generator.trainable_weights
        )
        discriminator_gradients = tape.gradient(
            discriminator_loss, self.discriminator.trainable_weights
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_weights)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_weights)
        )

        # 판별자의 성능에 따라 보강 확률을 업데이트
        self.augmenter.update(real_logits)

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        self.real_accuracy.update_state(1.0, step(real_logits))
        self.generated_accuracy.update_state(0.0, step(generated_logits))
        self.augmentation_probability_tracker.update_state(self.augmenter.probability)

        # 생성자의 가중치의 지수 이동 평균을 추적하여
        # 생성 품질의 분산을 줄입니다.
        for weight, ema_weight in zip(
            self.generator.weights, self.ema_generator.weights
        ):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID는 계산 효율성을 위해 트레이닝 단계에서는 측정되지 않습니다.
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, real_images):
        generated_images = self.generate(batch_size, training=False)

        self.kid.update_state(real_images, generated_images)

        # 계산 효율성을 위해 평가 단계에서는 KID만 측정됩니다.
        return {self.kid.name: self.kid.result()}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6, interval=5):
        # 생성 품질의 시각적 평가를 위해 랜덤하게 생성된 이미지를 플로팅합니다.
        if epoch is None or (epoch + 1) % interval == 0:
            num_images = num_rows * num_cols
            generated_images = self.generate(num_images, training=False)

            plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
            for row in range(num_rows):
                for col in range(num_cols):
                    index = row * num_cols + col
                    plt.subplot(num_rows, num_cols, index + 1)
                    plt.imshow(generated_images[index])
                    plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()
```

## 트레이닝 {#training}

트레이닝 중 메트릭을 보면, 실제 정확도(판별자가 실제 이미지에 대해 예측한 정확도)가
목표 정확도보다 낮으면 보강 확률이 증가하고,
그 반대의 경우에는 감소하는 것을 확인할 수 있습니다.
제 경험에 따르면, 건강한 GAN 트레이닝 동안 판별자의 정확도는 80-95% 범위에 있어야 합니다.
이 범위보다 낮으면 판별자가 너무 약하고, 높으면 너무 강한 것입니다.

생성자의 가중치의 지수 이동 평균을 추적하고,
이를 이미지 생성과 KID 평가에 사용한다는 점에 유의하세요.

```python
# 모델 생성 및 컴파일
model = GAN_ADA()
model.compile(
    generator_optimizer=keras.optimizers.Adam(learning_rate, beta_1),
    discriminator_optimizer=keras.optimizers.Adam(learning_rate, beta_1),
)

# 검증 KID 메트릭을 기준으로 최상의 모델을 저장
checkpoint_path = "gan_model"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)

# 트레이닝 실행 및 생성된 이미지를 주기적으로 플로팅
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
        checkpoint_callback,
    ],
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "generator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         [(None, 64)]              0
_________________________________________________________________
dense (Dense)                (None, 2048)              131072
_________________________________________________________________
batch_normalization (BatchNo (None, 2048)              6144
_________________________________________________________________
re_lu (ReLU)                 (None, 2048)              0
_________________________________________________________________
reshape (Reshape)            (None, 4, 4, 128)         0
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 8, 8, 128)         262144
_________________________________________________________________
batch_normalization_1 (Batch (None, 8, 8, 128)         384
_________________________________________________________________
re_lu_1 (ReLU)               (None, 8, 8, 128)         0
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 16, 16, 128)       262144
_________________________________________________________________
batch_normalization_2 (Batch (None, 16, 16, 128)       384
_________________________________________________________________
re_lu_2 (ReLU)               (None, 16, 16, 128)       0
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 32, 32, 128)       262144
_________________________________________________________________
batch_normalization_3 (Batch (None, 32, 32, 128)       384
_________________________________________________________________
re_lu_3 (ReLU)               (None, 32, 32, 128)       0
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 64, 64, 3)         6147
=================================================================
Total params: 930,947
Trainable params: 926,083
Non-trainable params: 4,864
_________________________________________________________________
Model: "discriminator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_3 (InputLayer)         [(None, 64, 64, 3)]       0
_________________________________________________________________
conv2d (Conv2D)              (None, 32, 32, 128)       6144
_________________________________________________________________
batch_normalization_4 (Batch (None, 32, 32, 128)       384
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 32, 32, 128)       0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 128)       262144
_________________________________________________________________
batch_normalization_5 (Batch (None, 16, 16, 128)       384
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 16, 16, 128)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 128)         262144
_________________________________________________________________
batch_normalization_6 (Batch (None, 8, 8, 128)         384
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 8, 8, 128)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 4, 128)         262144
_________________________________________________________________
batch_normalization_7 (Batch (None, 4, 4, 128)         384
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 4, 4, 128)         0
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0
_________________________________________________________________
dropout (Dropout)            (None, 2048)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 2049
=================================================================
Total params: 796,161
Trainable params: 795,137
Non-trainable params: 1,024
_________________________________________________________________
Epoch 1/10
46/46 [==============================] - 36s 307ms/step - g_loss: 3.3293 - d_loss: 0.1576 - real_acc: 0.9387 - gen_acc: 0.9579 - aug_p: 0.0020 - val_kid: 9.0999
Epoch 2/10
46/46 [==============================] - 10s 215ms/step - g_loss: 4.9824 - d_loss: 0.0912 - real_acc: 0.9704 - gen_acc: 0.9798 - aug_p: 0.0077 - val_kid: 8.3523
Epoch 3/10
46/46 [==============================] - 10s 218ms/step - g_loss: 5.0587 - d_loss: 0.1248 - real_acc: 0.9530 - gen_acc: 0.9625 - aug_p: 0.0131 - val_kid: 6.8116
Epoch 4/10
46/46 [==============================] - 10s 221ms/step - g_loss: 4.2580 - d_loss: 0.1002 - real_acc: 0.9686 - gen_acc: 0.9740 - aug_p: 0.0179 - val_kid: 5.2327
Epoch 5/10
46/46 [==============================] - 10s 225ms/step - g_loss: 4.6022 - d_loss: 0.0847 - real_acc: 0.9655 - gen_acc: 0.9852 - aug_p: 0.0234 - val_kid: 3.9004
```

![png](/images/examples/generative/gan_ada/gan_ada_18_3.png)

```plain
Epoch 6/10
46/46 [==============================] - 10s 224ms/step - g_loss: 4.9362 - d_loss: 0.0671 - real_acc: 0.9791 - gen_acc: 0.9895 - aug_p: 0.0291 - val_kid: 6.6020
Epoch 7/10
46/46 [==============================] - 10s 222ms/step - g_loss: 4.4272 - d_loss: 0.1184 - real_acc: 0.9570 - gen_acc: 0.9657 - aug_p: 0.0345 - val_kid: 3.3644
Epoch 8/10
46/46 [==============================] - 10s 220ms/step - g_loss: 4.5060 - d_loss: 0.1635 - real_acc: 0.9421 - gen_acc: 0.9594 - aug_p: 0.0392 - val_kid: 3.1381
Epoch 9/10
46/46 [==============================] - 10s 219ms/step - g_loss: 3.8264 - d_loss: 0.1667 - real_acc: 0.9383 - gen_acc: 0.9484 - aug_p: 0.0433 - val_kid: 2.9423
Epoch 10/10
46/46 [==============================] - 10s 219ms/step - g_loss: 3.4063 - d_loss: 0.1757 - real_acc: 0.9314 - gen_acc: 0.9475 - aug_p: 0.0473 - val_kid: 2.9112
```

![png](/images/examples/generative/gan_ada/gan_ada_18_5.png)

```plain
<keras.callbacks.History at 0x7fefcc2cb9d0>
```

{{% /details %}}

## 추론 {#inference}

```python
# 최상의 모델을 로드하고 이미지를 생성합니다.
model.load_weights(checkpoint_path)
model.plot_images()
```

![png](/images/examples/generative/gan_ada/gan_ada_20_0.png)

## 결과 {#results}

이 코드 예제를 사용하여 400 에포크 동안 트레이닝을 실행하면 (Colab 노트북에서 2-3시간 소요),
높은 품질의 이미지 생성을 얻을 수 있습니다.

400 에포크 트레이닝 동안 랜덤 배치의 이미지 진화 과정 (애니메이션 부드러움을 위해 ema=0.999):

![birds evolution gif](/images/examples/generative/gan_ada/ecGuCcz.gif)

선택된 이미지 배치 사이의 잠재 공간 보간:

![birds evolution gif](/images/examples/generative/gan_ada/nGvzlsC.gif)

다른 데이터셋, 예를 들어 [CelebA](https://www.tensorflow.org/datasets/catalog/celeb_a)로
트레이닝을 시도해보는 것도 추천합니다.
제 경험으로는 어떤 하이퍼파라미터도 변경하지 않고도 좋은 결과를 얻을 수 있습니다.
(다만, 판별자 보강이 필요하지 않을 수 있습니다)

## GAN 팁 및 트릭 {#gan-tips-and-tricks}

이 예제의 목표는 GAN에 대한 구현의 용이성과 생성 품질 사이에서 좋은 균형을 찾는 것이었습니다.
준비 과정에서, 저는 [이 저장소](https://github.com/beresandras/gan-flavours-keras)를 사용하여,
수많은 절제 실험(ablations)을 수행했습니다.

이 섹션에서는 제가 배운 교훈과 주관적으로 중요하다고 생각하는 순서대로 제 추천 사항을 나열합니다.

또한, [DCGAN 논문](https://arxiv.org/abs/1511.06434),
이 [NeurIPS 강연](https://www.youtube.com/watch?v=myGAju4L7O8),
그리고 이 [대규모 GAN 연구](https://arxiv.org/abs/1711.10337)를 참고하여,
다른 연구자들의 의견도 확인해 보시기 바랍니다.

### 아키텍쳐 팁 {#architectural-tips}

- **해상도**: 더 높은 해상도에서 GAN을 트레이닝하는 것은 점점 더 어려워지기 때문에,
  처음에는 32x32 또는 64x64 해상도에서 실험해 볼 것을 권장합니다.
- **초기화**: 트레이닝 초기에 강한 컬러 패턴이 나타나면, 초기화가 문제일 수 있습니다.
  레이어의 `kernel_initializer` 파라미터를 [랜덤 노말]({{< relref "/docs/api/layers/initializers/#randomnormal-class" >}})로 설정하고, 표준 편차를 감소시키세요. (DCGAN을 따르며 추천 값: 0.02)
  이 값이 문제를 해결할 때까지 조정하세요.
- **업샘플링**: 생성자에서 업샘플링을 위한 두 가지 메인 메서드가 있습니다.
  [전치 컨볼루션]({{< relref "/docs/api/layers/convolution_layers/convolution2d_transpose" >}})은 더 빠르지만,
  [체커보드 아티팩트](https://distill.pub/2016/deconv-checkerboard/)를 유발할 수 있습니다.
  이를 줄이기 위해서는 stride와 나누어 떨어지는 커널 크기를 사용하는 것이 좋습니다. (stride가 2일 때 추천 커널 크기는 4)
  [업샘플링]({{< relref "/docs/api/layers/reshaping_layers/up_sampling2d" >}}) + [표준 컨볼루션]({{< relref "/docs/api/layers/convolution_layers/convolution2d" >}})은 품질이 약간 낮을 수 있지만, 체커보드 아티팩트 문제는 없습니다.
  이를 위해서는 최근접 이웃 보간을 선형 보간 대신 사용하는 것이 좋습니다.
- **판별자에서의 배치 정규화**: 때로는 큰 영향을 미칠 수 있으므로, 두 가지 방법을 모두 시도해 볼 것을 권장합니다.
- **[스펙트럼 정규화](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/SpectralNormalization)**:
  GAN 트레이닝을 위한 인기 있는 기법으로, 안정성에 도움이 될 수 있습니다.
  이를 사용할 때 배치 정규화의 학습 가능한 스케일 파라미터를 비활성화할 것을 권장합니다.
- **[residual 연결]({{< relref "/docs/guides/functional_api/#a-toy-resnet-model" >}})**:
  residual 판별자는 유사하게 작동하지만, 제 경험상 residual 생성자는 트레이닝이 더 어렵습니다.
  그러나 대규모 깊은 아키텍처를 트레이닝할 때는 필수적입니다.
  residual 연결이 없는 아키텍처로 시작하는 것을 권장합니다.
- **드롭아웃**: 판별자의 마지막 레이어 전에 드롭아웃을 사용하면 생성 품질이 향상된다는 것이 제 경험입니다.
  추천 드롭아웃 비율은 0.5 이하입니다.
- **[Leaky ReLU]({{< relref "/docs/api/layers/activation_layers/leaky_relu" >}})**:
  판별자에서 Leaky ReLU 활성화를 사용하여 그래디언트를 덜 희소하게(less sparse) 만드세요.
  DCGAN을 따라 추천 기울기/알파는 0.2입니다.

### 알고리즘 팁 {#algorithmic-tips}

- **손실 함수**: GAN 트레이닝을 위해 여러 해에 걸쳐 많은 손실 함수가 제안되었으며, 더 나은 성능과 안정성을 약속합니다.
  [이 저장소](https://github.com/beresandras/gan-flavours-keras)에서 5가지 손실 함수를 구현했으며,
  제 경험은 [이 GAN 연구](https://arxiv.org/abs/1711.10337)와 일치합니다.
  기본 비포화 GAN 손실을 일관되게 능가하는 손실 함수는 없어 보입니다.
  기본값으로 이를 사용하는 것을 권장합니다.
- **Adam의 beta_1 파라미터**: Adam의 beta_1 파라미터는 평균 그래디언트 추정의 모멘텀으로 해석될 수 있습니다.
  기본값인 0.9 대신 0.5 또는 심지어 0.0을 사용하는 것이 DCGAN에서 제안되었으며, 이는 중요합니다.
  이 예제는 기본값을 사용할 경우, 작동하지 않을 것입니다.
- **생성된 이미지와 실제 이미지에 대한 별도의 배치 정규화**:
  판별자의 순전파는 생성된 이미지와 실제 이미지에 대해 별도로 수행해야 합니다.
  그렇지 않으면, 아티팩트(제 경우에는 45도 줄무늬)가 나타나고 성능이 저하될 수 있습니다.
- **생성자의 가중치의 지수 이동 평균**:
  이는 KID 측정의 분산을 줄이는 데 도움을 주고,
  트레이닝 중 급격한 색상 팔레트 변화를 평균화하는 데 도움이 됩니다.
- **[생성자와 판별자에 대한 다른 학습률](https://arxiv.org/abs/1706.08500)**:
  리소스가 충분하다면, 두 네트워크의 학습률을 따로 조정하는 것이 도움이 될 수 있습니다.
  비슷한 아이디어로는, 한 네트워크(보통 판별자)의 가중치를 다른 네트워크의 업데이트마다 여러 번 업데이트하는 것입니다.
  두 네트워크 모두 DCGAN을 따라 2e-4 (Adam) 학습률을 사용하고,
  기본값으로 두 네트워크를 한 번만 업데이트하는 것을 권장합니다.
- **레이블 노이즈**: [일방향 레이블 스무딩](https://arxiv.org/abs/1606.03498) (실제 레이블에 대해 1.0 미만을 사용) 또는
  레이블에 노이즈를 추가하는 것은 판별자가 과도하게 확신하지 않도록 정규화할 수 있습니다.
  그러나 제 경우에는 성능을 개선하지 않았습니다.
- **적응형 데이터 보강**: 트레이닝 프로세스에 또 다른 동적 요소를 추가하기 때문에,
  기본적으로 비활성화하고 다른 요소들이 이미 잘 작동할 때만 활성화하는 것이 좋습니다.

## 관련 연구 {#related-works}

다른 GAN 관련 Keras 코드 예제:

- [DCGAN + CelebA]({{< relref "/docs/examples/generative/dcgan_overriding_train_step" >}})
- [WGAN + FashionMNIST]({{< relref "/docs/examples/generative/wgan_gp" >}})
- [WGAN + Molecules]({{< relref "/docs/examples/generative/wgan-graphs" >}})
- [ConditionalGAN + MNIST]({{< relref "/docs/examples/generative/conditional_gan" >}})
- [CycleGAN + Horse2Zebra]({{< relref "/docs/examples/generative/cyclegan" >}})
- [StyleGAN]({{< relref "/docs/examples/generative/stylegan" >}})

최신 GAN 아키텍처 라인:

- [SAGAN](https://arxiv.org/abs/1805.08318), [BigGAN](https://arxiv.org/abs/1809.11096)
- [ProgressiveGAN](https://arxiv.org/abs/1710.10196), [StyleGAN](https://arxiv.org/abs/1812.04948), [StyleGAN2](https://arxiv.org/abs/1912.04958), [StyleGAN2-ADA](https://arxiv.org/abs/2006.06676), [AliasFreeGAN](https://arxiv.org/abs/2106.12423)

판별자 데이터 보강에 관한 동시 논문: [1](https://arxiv.org/abs/2006.02595), [2](https://arxiv.org/abs/2006.05338), [3](https://arxiv.org/abs/2006.10738)

최근 GAN에 대한 문헌 개요: [강연](https://www.youtube.com/watch?v=3ktD752xq5k)
