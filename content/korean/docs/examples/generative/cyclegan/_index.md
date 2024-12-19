---
title: CycleGAN
linkTitle: CycleGAN
toc: true
weight: 11
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** [A_K_Nain](https://twitter.com/A_K_Nain)  
**{{< t f_date_created >}}** 2020/08/12  
**{{< t f_last_modified >}}** 2024/09/30  
**{{< t f_description >}}** CycleGAN 구현.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/cyclegan.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/cyclegan.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## CycleGAN {#cyclegan}

CycleGAN은 이미지-투-이미지 변환 문제를 해결하려는 모델입니다.
이미지-투-이미지 변환 문제의 목표는 정렬된 이미지 쌍으로 구성된 트레이닝 세트를 사용하여,
입력 이미지와 출력 이미지 간의 매핑을 학습하는 것입니다.
하지만, 매칭된 예시를 얻는 것은 항상 가능한 일이 아닙니다.
CycleGAN은 페어링된 입력-출력 이미지 없이 이 매핑을 학습하려고 시도하며,
사이클-일관성 적대적 신경망(cycle-consistent adversarial networks)을 사용합니다.

- [논문](https://arxiv.org/abs/1703.10593)
- [원본 구현](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## 셋업 {#setup}

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, ops
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
autotune = tf.data.AUTOTUNE

os.environ["KERAS_BACKEND"] = "tensorflow"
```

## 데이터셋 준비 {#prepare-the-dataset}

이 예제에서는 [horse to zebra](https://www.tensorflow.org/datasets/catalog/cycle_gan#cycle_ganhorse2zebra) 데이터셋을 사용할 것입니다.

```python
# TensorFlow 데이터셋을 사용하여 horse-zebra 데이터셋을 로드합니다.
dataset, _ = tfds.load(name="cycle_gan/horse2zebra", with_info=True, as_supervised=True)
train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
test_horses, test_zebras = dataset["testA"], dataset["testB"]

# 표준 이미지 크기를 정의합니다.
orig_img_size = (286, 286)
# 트레이닝 동안 사용할 랜덤 크롭의 크기입니다.
input_img_size = (256, 256, 3)
# 레이어의 가중치 초기화 함수입니다.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# 인스턴스 정규화를 위한 감마 초기화 함수입니다.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

buffer_size = 256
batch_size = 1


def normalize_img(img):
    img = ops.cast(img, dtype=tf.float32)
    # 값을 [-1, 1] 범위로 매핑합니다.
    return (img / 127.5) - 1.0


def preprocess_train_image(img, label):
    # 랜덤 플립
    img = tf.image.random_flip_left_right(img)
    # 먼저 원래 크기로 리사이즈합니다.
    img = ops.image.resize(img, [*orig_img_size])
    # 256X256 크기로 랜덤 크롭합니다.
    img = tf.image.random_crop(img, size=[*input_img_size])
    # 픽셀 값을 [-1, 1] 범위로 정규화합니다.
    img = normalize_img(img)
    return img


def preprocess_test_image(img, label):
    # 테스트 이미지의 경우, 리사이즈와 정규화만 수행합니다.
    img = ops.image.resize(img, [input_img_size[0], input_img_size[1]])
    img = normalize_img(img)
    return img
```

## `Dataset` 객체 만들기 {#create-dataset-objects}

```python
# 트레이닝 데이터에 전처리 작업을 적용합니다.
train_horses = (
    train_horses.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)
train_zebras = (
    train_zebras.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)

# 테스트 데이터에 전처리 작업을 적용합니다.
test_horses = (
    test_horses.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)
test_zebras = (
    test_zebras.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)
```

## 일부 샘플 시각화 {#visualize-some-samples}

```python
_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, samples in enumerate(zip(train_horses.take(4), train_zebras.take(4))):
    horse = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    zebra = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    ax[i, 0].imshow(horse)
    ax[i, 1].imshow(zebra)
plt.show()
```

![png](/images/examples/generative/cyclegan/cyclegan_9_0.png)

## CycleGAN 생성자와 판별자에 사용된 빌딩 블록 {#building-blocks-used-in-the-cyclegan-generators-and-discriminators}

```python
class ReflectionPadding2D(layers.Layer):
    """Reflection Padding을 레이어로 구현합니다.

    Args:
        padding(tuple): 공간 차원에 대한 패딩 양.

    Returns:
        입력 텐서와 동일한 타입의 패딩된 텐서.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return ops.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = keras.layers.GroupNormalization(groups=1, gamma_initializer=gamma_initializer)(
        x
    )
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = keras.layers.GroupNormalization(groups=1, gamma_initializer=gamma_initializer)(
        x
    )
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = keras.layers.GroupNormalization(groups=1, gamma_initializer=gamma_initializer)(
        x
    )
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = keras.layers.GroupNormalization(groups=1, gamma_initializer=gamma_initializer)(
        x
    )
    if activation:
        x = activation(x)
    return x
```

## 생성자 빌드 {#build-the-generators}

생성자는 다운샘플링 블록, 9개의 residual 블록, 그리고 업샘플링 블록으로 구성됩니다.
생성자의 구조는 다음과 같습니다:

```plain
c7s1-64 ==> `relu` 활성화 함수와 필터 크기 7을 가진 Conv 블록
d128 ====|
         |-> 2개의 다운샘플링 블록
d256 ====|
R256 ====|
R256     |
R256     |
R256     |
R256     |-> 9개의 residual 블록
R256     |
R256     |
R256     |
R256 ====|
u128 ====|
         |-> 2개의 업샘플링 블록
u64  ====|
c7s1-3 => `tanh` 활성화 함수와 필터 크기 7을 가진 마지막 Conv 블록
```

```python
def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
        x
    )
    x = keras.layers.GroupNormalization(groups=1, gamma_initializer=gamma_initializer)(
        x
    )
    x = layers.Activation("relu")(x)

    # 다운샘플링
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual 블록
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # 업샘플링
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # 마지막 블록
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model
```

## 판별자 빌드 {#build-the-discriminators}

판별자는 다음과 같은 아키텍처를 구현합니다: `C64->C128->C256->C512`

```python
def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


# 생성자 가져오기
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# 판별자 가져오기
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")
```

## CycleGAN 모델 빌드 {#build-the-cyclegan-model}

`fit()`을 통해 트레이닝하기 위해 `Model` 클래스의 `train_step()` 메서드를 재정의하겠습니다.

```python
class CycleGan(keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super().__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def call(self, inputs):
        return (
            self.disc_X(inputs),
            self.disc_Y(inputs),
            self.gen_G(inputs),
            self.gen_F(inputs),
        )

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super().compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x는 말 이미지, y는 얼룩말 이미지입니다.
        real_x, real_y = batch_data

        # CycleGAN에서는, 생성자와 판별자에 대해
        # 서로 다른 손실을 계산해야 합니다.
        # 여기서는 다음 단계를 수행합니다:
        #
        # 1. 실제 이미지를 생성자에 통과시켜 생성된 이미지를 얻습니다.
        # 2. 생성된 이미지를 다시 생성자에 통과시켜,
        #    생성된 이미지로부터 원본 이미지를 예측할 수 있는지 확인합니다.
        # 3. 생성자를 사용하여 실제 이미지의 동일성 매핑(identity mapping)을 수행합니다.
        # 4. 1)에서 생성된 이미지를 해당 판별자에 통과시킵니다.
        # 5. 생성자의 총 손실(적대적(adversarial) + 사이클 + 동일성(identity))을 계산합니다.
        # 6. 판별자의 손실을 계산합니다.
        # 7. 생성자의 가중치를 업데이트합니다.
        # 8. 판별자의 가중치를 업데이트합니다.
        # 9. 손실을 딕셔너리 형태로 반환합니다.

        with tf.GradientTape(persistent=True) as tape:
            # '말'에서 '가짜 얼룩말'으로
            fake_y = self.gen_G(real_x, training=True)
            # '얼룩말'에서 '가짜 말'로 -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # 사이클 (말 -> 가짜 얼룩말 -> 가짜 말): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # 사이클 (얼룩말 -> 가짜 말 -> 가짜 얼룩말) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # 동일성 매핑 (Identity mapping)
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # 판별자 출력
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # 생성자 적대적 손실
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # 생성자 사이클 손실
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # 생성자 동일성 손실
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # 생성자의 총 손실
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # 판별자 손실
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # 생성자의 그래디언트 구하기
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # 판별자의 그래디언트 구하기
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # 생성자의 가중치 업데이트
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # 판별자의 가중치 업데이트
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }
```

## 생성된 이미지를 주기적으로 저장하는 콜백 만들기 {#create-a-callback-that-periodically-saves-generated-images}

```python
class GANMonitor(keras.callbacks.Callback):
    """매 에포크 후 이미지를 생성하고 저장하는 콜백"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(test_horses.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.utils.array_to_img(prediction)
            prediction.save(
                "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            )
        plt.show()
        plt.close()
```

## 엔드투엔드 모델 트레이닝 {#train-the-end-to-end-model}

```python
# 적대적 손실을 평가하기 위한 손실 함수
adv_loss_fn = keras.losses.MeanSquaredError()


# 생성자를 위한 손실 함수 정의
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(ops.ones_like(fake), fake)
    return fake_loss


# 판별자를 위한 손실 함수 정의
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(ops.ones_like(real), real)
    fake_loss = adv_loss_fn(ops.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


# Cycle GAN 모델 생성
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# 모델 컴파일
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)
# 콜백 설정
plotter = GANMonitor()
checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.weights.h5"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True
)

# 각 에포크가 단일 P100 머신에서 약 7분이 걸리므로,
# 여기서는 모델을 1 에포크만 트레이닝합니다.
cycle_gan_model.fit(
    tf.data.Dataset.zip((train_horses, train_zebras)),
    epochs=90,
    callbacks=[plotter, model_checkpoint_callback],
)
```

모델의 성능을 테스트합니다.

```python
# 가중치가 로드되면, 테스트 데이터에서 몇 가지 샘플을 가져와 모델 성능을 확인합니다.


# 체크포인트를 로드합니다.
cycle_gan_model.load_weights(checkpoint_filepath)
print("Weights loaded successfully")

_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, img in enumerate(test_horses.take(4)):
    prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input image")
    ax[i, 0].set_title("Input image")
    ax[i, 1].set_title("Translated image")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")

    prediction = keras.utils.array_to_img(prediction)
    prediction.save("predicted_img_{i}.png".format(i=i))
plt.tight_layout()
plt.show()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Weights loaded successfully
```

{{% /details %}}

![png](/images/examples/generative/cyclegan/cyclegan_23_1.png)
