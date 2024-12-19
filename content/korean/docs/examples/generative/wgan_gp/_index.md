---
title: "`Model.train_step`을 오버라이딩 하는 WGAN-GP"
linkTitle: WGAN-GP
toc: true
weight: 9
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** [A_K_Nain](https://twitter.com/A_K_Nain)  
**{{< t f_date_created >}}** 2020/05/09  
**{{< t f_last_modified >}}** 2023/08/03  
**{{< t f_description >}}** Gradient Penalty를 사용한 Wasserstein GAN 구현.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/wgan_gp.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/wgan_gp.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Gradient Penalty(GP)를 갖춘 Wasserstein GAN(WGAN) {#wasserstein-gan-wgan-with-gradient-penalty-gp}

원본 [Wasserstein GAN](https://arxiv.org/abs/1701.07875)은 Wasserstein 거리를 활용하여,
원본 GAN 논문에서 사용된 값 함수(value function)보다 더 나은 이론적 속성을 가진 값 함수를 생성합니다.
WGAN은 판별자(discriminator, 일명 비평가(critic))가 1-Lipschitz 함수 공간 내에 있어야 합니다.
저자는 이 제약 조건을 달성하기 위해, 가중치 클리핑(weight clipping)이라는 아이디어를 제안했습니다.
가중치 클리핑은 작동하지만, 1-Lipschitz 제약 조건을 적용하는 데 문제가 될 수 있으며,
바람직하지 않은 동작을 일으킬 수 있습니다.
예를 들어, 매우 깊은 WGAN 판별자(비평가)는 종종 수렴하지 못합니다.

[WGAN-GP](https://arxiv.org/abs/1704.00028) 방법은 원활한 트레이닝을 ​​보장하기 위해,
가중치 클리핑에 대한 대안을 제안합니다.
저자는 가중치를 클리핑하는 대신,
판별자 그래디언트의 L2 norm을 1에 가깝게 유지하는 손실 항을 추가하는,
"그래디언트 페널티(Gradient Penalty, GP)"를 제안했습니다.

## 셋업 {#setup}

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from keras import layers
```

## Fashion-MNIST 데이터 준비 {#prepare-the-fashion-mnist-data}

WGAN-GP를 트레이닝하는 방법을 보여주기 위해,
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) 데이터 세트를 사용합니다.
이 데이터 세트의 각 샘플은 10개 클래스(예: 바지, 풀오버, 스니커즈 등)의 레이블과 연결된 28x28 회색조 이미지입니다.

```python
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 512

# 노이즈 벡터의 크기
noise_dim = 128

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(f"Number of examples: {len(train_images)}")
print(f"Shape of the images in the dataset: {train_images.shape[1:]}")

# 각 샘플을 (28, 28, 1)로 reshape하고,
# [-1, 1] 범위로 픽셀 값을 정규화합니다.
train_images = train_images.reshape(train_images.shape[0], *IMG_SHAPE).astype("float32")
train_images = (train_images - 127.5) / 127.5
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
 29515/29515 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
 26421880/26421880 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
 5148/5148 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
 4422102/4422102 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Number of examples: 60000
Shape of the images in the dataset: (28, 28)
```

{{% /details %}}

## 판별자(원본 WGAN의 비평가) 만들기 {#create-the-discriminator-the-critic-in-the-original-wgan}

데이터 세트의 샘플은 (28, 28, 1) 모양을 갖습니다.
스트라이드된 컨볼루션을 사용할 것이므로, 홀수 차원의 모양이 생길 수 있습니다.
예를 들어, `(28, 28) -> Conv_s2 -> (14, 14) -> Conv_s2 -> (7, 7) -> Conv_s2 ->(3, 3)`.

네트워크의 생성자 부분에서 업샘플링을 수행하는 동안,
주의하지 않으면 원본 이미지와 동일한 입력 모양을 얻지 못합니다.
이를 방지하기 위해 훨씬 간단한 작업을 수행합니다.

- 판별자에서: 입력을 "제로 패딩"하여 각 샘플의 모양을 `(32, 32, 1)`로 변경합니다.
- 생성자에서: 입력 모양과 모양이 일치하도록 최종 출력을 자릅니다.

```python
def conv_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=0.5,
):
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_discriminator_model():
    img_input = layers.Input(shape=IMG_SHAPE)
    # 입력 이미지 크기를 (32, 32, 1)로 만들기 위해, 입력을 0으로 채웁니다. (제로 패딩)
    x = layers.ZeroPadding2D((2, 2))(img_input)
    x = conv_block(
        x,
        64,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(0.2),
        use_dropout=False,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        128,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        256,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        512,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
    )

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)

    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


d_model = get_discriminator_model()
d_model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "discriminator"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 28, 28, 1)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ zero_padding2d (ZeroPadding2D)  │ (None, 32, 32, 1)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d (Conv2D)                 │ (None, 16, 16, 64)        │      1,664 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu (LeakyReLU)         │ (None, 16, 16, 64)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (Conv2D)               │ (None, 8, 8, 128)         │    204,928 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_1 (LeakyReLU)       │ (None, 8, 8, 128)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (Dropout)               │ (None, 8, 8, 128)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_2 (Conv2D)               │ (None, 4, 4, 256)         │    819,456 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_2 (LeakyReLU)       │ (None, 4, 4, 256)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout_1 (Dropout)             │ (None, 4, 4, 256)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_3 (Conv2D)               │ (None, 2, 2, 512)         │  3,277,312 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_3 (LeakyReLU)       │ (None, 2, 2, 512)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ flatten (Flatten)               │ (None, 2048)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout_2 (Dropout)             │ (None, 2048)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (None, 1)                 │      2,049 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 4,305,409 (16.42 MB)
 Trainable params: 4,305,409 (16.42 MB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

## 생성자 만들기 {#create-the-generator}

```python
def upsample_block(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    up_size=(2, 2),
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=0.3,
):
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_generator_model():
    noise = layers.Input(shape=(noise_dim,))
    x = layers.Dense(4 * 4 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((4, 4, 256))(x)
    x = upsample_block(
        x,
        128,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        64,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x, 1, layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
    )
    # 이 시점에서 우리는 입력과 같은 모양(32, 32, 1)을 가진 출력을 얻습니다.
    # 우리는 그것을 (28, 28, 1)로 만들기 위해, Cropping2D 레이어를 사용할 것입니다.
    x = layers.Cropping2D((2, 2))(x)

    g_model = keras.models.Model(noise, x, name="generator")
    return g_model


g_model = get_generator_model()
g_model.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "generator"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)      │ (None, 128)               │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (Dense)                 │ (None, 4096)              │    524,288 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ batch_normalization             │ (None, 4096)              │     16,384 │
│ (BatchNormalization)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_4 (LeakyReLU)       │ (None, 4096)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ reshape (Reshape)               │ (None, 4, 4, 256)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ up_sampling2d (UpSampling2D)    │ (None, 8, 8, 256)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_4 (Conv2D)               │ (None, 8, 8, 128)         │    294,912 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ batch_normalization_1           │ (None, 8, 8, 128)         │        512 │
│ (BatchNormalization)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_5 (LeakyReLU)       │ (None, 8, 8, 128)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ up_sampling2d_1 (UpSampling2D)  │ (None, 16, 16, 128)       │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_5 (Conv2D)               │ (None, 16, 16, 64)        │     73,728 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ batch_normalization_2           │ (None, 16, 16, 64)        │        256 │
│ (BatchNormalization)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_6 (LeakyReLU)       │ (None, 16, 16, 64)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ up_sampling2d_2 (UpSampling2D)  │ (None, 32, 32, 64)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_6 (Conv2D)               │ (None, 32, 32, 1)         │        576 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ batch_normalization_3           │ (None, 32, 32, 1)         │          4 │
│ (BatchNormalization)            │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ activation (Activation)         │ (None, 32, 32, 1)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ cropping2d (Cropping2D)         │ (None, 28, 28, 1)         │          0 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 910,660 (3.47 MB)
 Trainable params: 902,082 (3.44 MB)
 Non-trainable params: 8,578 (33.51 KB)
```

{{% /details %}}

## WGAN-GP 모델 생성 {#create-the-wgan-gp-model}

이제 생성자와 판별자를 정의했으니, WGAN-GP 모델을 구현할 차례입니다.
또한 트레이닝을 ​​위해 `train_step`을 재정의합니다.

```python
class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """그래디언트 페널티(gradient penalty, GP)를 계산합니다.

        이 손실은 보간된 이미지에서 계산되어, 판별자 손실에 추가됩니다.
        """
        # 보간된 이미지를 얻습니다.
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. 이 보간된 이미지에 대한 판별자 출력을 가져옵니다.
            pred = self.discriminator(interpolated, training=True)

        # 2. 이 보간된 이미지에 대한 그래디언트를 계산합니다.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. 그래디언트의 norm을 계산합니다.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # 배치 크기를 가져옵니다.
        batch_size = tf.shape(real_images)[0]

        # 각 배치에 대해 원본 논문에 나와 있는 대로 다음 단계를 수행합니다.
        # 1. 생성자를 트레이닝하고, 생성자 손실을 구합니다.
        # 2. 판별자를 트레이닝하고, 판별자 손실을 구합니다.
        # 3. 그래디언트 페널티(GP)를 계산합니다.
        # 4. 이 그래디언트 페널티를 상수 가중치 인자(factor)와 곱합니다.
        # 5. 판별자 손실에 그래디언트 페널티를 추가합니다.
        # 6. 생성자 및 판별자 손실을 손실 딕셔너리로 반환합니다.

        # 먼저 판별자를 트레이닝합니다.
        # 원본 논문에서는 생성자의 한 단계보다 판별자를 `x` 단계 더(일반적으로 5) 트레이닝하는 것을 권장합니다.
        # 여기서는 트레이닝 시간을 줄이기 위해, 5단계가 아닌 3단계를 더 트레이닝합니다.
        for i in range(self.d_steps):
            # 잠재 벡터를 얻습니다.
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # 잠재 벡터에서 가짜 이미지 생성
                fake_images = self.generator(random_latent_vectors, training=True)
                # 가짜 이미지에 대한 로짓을 얻습니다
                fake_logits = self.discriminator(fake_images, training=True)
                # 실제 이미지에 대한 로짓을 얻습니다
                real_logits = self.discriminator(real_images, training=True)

                # 가짜 및 실제 이미지 로짓을 사용하여, 판별자 손실을 계산합니다.
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # 그래디언트 페널티를 계산합니다
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # 원래 판별자 손실에 그래디언트 페널티를 추가합니다.
                d_loss = d_cost + gp * self.gp_weight

            # 판별자 손실에 대한 그래디언트를 얻습니다.
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # 판별자 옵티마이저를 사용하여, 판별자의 가중치를 업데이트합니다.
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # 생성자를 트레이닝합니다.
        # 잠재 벡터를 얻습니다.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # 생성기를 사용하여 가짜 이미지 생성
            generated_images = self.generator(random_latent_vectors, training=True)
            # 가짜 이미지에 대한 판별자 로짓을 얻습니다.
            gen_img_logits = self.discriminator(generated_images, training=True)
            # 생성자 손실을 계산합니다.
            g_loss = self.g_loss_fn(gen_img_logits)

        # 생성자 손실에 대한 그래디언트를 얻습니다.
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # 생성자 옵티마이저를 사용하여 생성자의 가중치를 업데이트합니다.
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}
```

## 생성된 이미지를 주기적으로 저장하는 Keras 콜백 만들기 {#create-a-keras-callback-that-periodically-saves-generated-images}

```python
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.utils.array_to_img(img)
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
```

## 엔드투엔드 모델 트레이닝 {#train-the-end-to-end-model}

```python
# 두 네트워크 모두에 대한 옵티마이저를 인스턴스화합니다.
# (learning_rate=0.0002, beta_1=0.5가 권장됩니다)
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)


# 판별자에 대한 손실 함수를 정의합니다.
# 이는 (fake_loss - real_loss)여야 합니다.
# 나중에 이 손실 함수에 그래디언트 페널티를 추가합니다.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# 생성자의 손실 함수를 정의합니다.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


# 트레이닝을 위한 에포크 수를 설정합니다.
epochs = 20

# customer `GANMonitor` Keras 콜백을 인스턴스화합니다.
cbk = GANMonitor(num_img=3, latent_dim=noise_dim)

# Wgan 모델을 얻습니다.
wgan = WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=noise_dim,
    discriminator_extra_steps=3,
)

# wgan 모델을 컴파일합니다.
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

# 트레이닝 시작
wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 79s 345ms/step - d_loss: -7.7597 - g_loss: -17.2858 - loss: 0.0000e+00
Epoch 2/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 118ms/step - d_loss: -7.0841 - g_loss: -13.8542 - loss: 0.0000e+00
Epoch 3/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 118ms/step - d_loss: -6.1011 - g_loss: -13.2763 - loss: 0.0000e+00
Epoch 4/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 119ms/step - d_loss: -5.5292 - g_loss: -13.3122 - loss: 0.0000e+00
Epoch 5/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 119ms/step - d_loss: -5.1012 - g_loss: -12.1395 - loss: 0.0000e+00
Epoch 6/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 119ms/step - d_loss: -4.7557 - g_loss: -11.2559 - loss: 0.0000e+00
Epoch 7/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 119ms/step - d_loss: -4.4727 - g_loss: -10.3075 - loss: 0.0000e+00
Epoch 8/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 119ms/step - d_loss: -4.2056 - g_loss: -10.0340 - loss: 0.0000e+00
Epoch 9/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -4.0116 - g_loss: -9.9283 - loss: 0.0000e+00
Epoch 10/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -3.8050 - g_loss: -9.7392 - loss: 0.0000e+00
Epoch 11/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -3.6608 - g_loss: -9.4686 - loss: 0.0000e+00
Epoch 12/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 121ms/step - d_loss: -3.4623 - g_loss: -8.9601 - loss: 0.0000e+00
Epoch 13/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -3.3659 - g_loss: -8.4620 - loss: 0.0000e+00
Epoch 14/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -3.2486 - g_loss: -7.9598 - loss: 0.0000e+00
Epoch 15/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -3.1436 - g_loss: -7.5392 - loss: 0.0000e+00
Epoch 16/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -3.0370 - g_loss: -7.3694 - loss: 0.0000e+00
Epoch 17/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -2.9256 - g_loss: -7.6105 - loss: 0.0000e+00
Epoch 18/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -2.8976 - g_loss: -6.5240 - loss: 0.0000e+00
Epoch 19/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -2.7944 - g_loss: -6.6281 - loss: 0.0000e+00
Epoch 20/20
 118/118 ━━━━━━━━━━━━━━━━━━━━ 14s 120ms/step - d_loss: -2.7175 - g_loss: -6.5900 - loss: 0.0000e+00

<keras.src.callbacks.history.History at 0x7fc763a8e950>
```

{{% /details %}}

마지막으로 생성된 이미지 표시:

```python
from IPython.display import Image, display

display(Image("generated_img_0_19.png"))
display(Image("generated_img_1_19.png"))
display(Image("generated_img_2_19.png"))
```

![png](/images/examples/generative/wgan_gp/wgan_gp_17_0.png)

![png](/images/examples/generative/wgan_gp/wgan_gp_17_1.png)

![png](/images/examples/generative/wgan_gp/wgan_gp_17_2.png)
