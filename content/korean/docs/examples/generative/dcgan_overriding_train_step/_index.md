---
title: 얼굴 이미지 생성을 위한 DCGAN
linkTitle: DCGAN 얼굴 이미지 생성
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2019/04/29  
**{{< t f_last_modified >}}** 2023/12/21  
**{{< t f_description >}}** `fit()`를 사용하여 CelebA 이미지에 대해 `train_step`을 재정의하여 트레이닝한 간단한 DCGAN입니다.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/dcgan_overriding_train_step.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/dcgan_overriding_train_step.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 셋업 {#setup}

```python
import keras
import tensorflow as tf

from keras import layers
from keras import ops
import matplotlib.pyplot as plt
import os
import gdown
from zipfile import ZipFile
```

## CelebA 데이터 준비 {#prepare-celeba-data}

CelebA 데이터세트의 얼굴 이미지를 64x64로 조정하여 사용하겠습니다.

```python
os.makedirs("celeba_gan")

url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
output = "celeba_gan/data.zip"
gdown.download(url, output, quiet=True)

with ZipFile("celeba_gan/data.zip", "r") as zipobj:
    zipobj.extractall("celeba_gan")
```

폴더에서 데이터 세트를 만들고, 이미지 크기를 \[0-1\] 범위로 조정합니다.

```python
dataset = keras.utils.image_dataset_from_directory(
    "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=32
)
dataset = dataset.map(lambda x: x / 255.0)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Found 202599 files.
```

{{% /details %}}

샘플 이미지를 표시해 보겠습니다.

```python
for x in dataset:
    plt.axis("off")
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    break
```

![png](/images/examples/generative/dcgan_overriding_train_step/dcgan_overriding_train_step_8_0.png)

## 판별자(discriminator) 만들기 {#create-the-discriminator}

64x64 이미지를 이진 분류 점수에 매핑합니다.

```python
discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "discriminator"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 32, 32, 64)        │      3,136 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu (LeakyReLU)         │ (None, 32, 32, 64)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (Conv2D)               │ (None, 16, 16, 128)       │    131,200 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_1 (LeakyReLU)       │ (None, 16, 16, 128)       │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_2 (Conv2D)               │ (None, 8, 8, 128)         │    262,272 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_2 (LeakyReLU)       │ (None, 8, 8, 128)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ flatten (Flatten)               │ (None, 8192)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (Dropout)               │ (None, 8192)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (None, 1)                 │      8,193 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 404,801 (1.54 MB)
 Trainable params: 404,801 (1.54 MB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

## 생성자(generator) 만들기 {#create-the-generator}

이는 판별자를 반영하여, `Conv2D` 레이어를 `Conv2DTranspose` 레이어로 대체했습니다.

```python
latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(8 * 8 * 128),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "generator"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ dense_1 (Dense)                 │ (None, 8192)              │  1,056,768 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ reshape (Reshape)               │ (None, 8, 8, 128)         │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_transpose                │ (None, 16, 16, 128)       │    262,272 │
│ (Conv2DTranspose)               │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_3 (LeakyReLU)       │ (None, 16, 16, 128)       │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_transpose_1              │ (None, 32, 32, 256)       │    524,544 │
│ (Conv2DTranspose)               │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_4 (LeakyReLU)       │ (None, 32, 32, 256)       │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_transpose_2              │ (None, 64, 64, 512)       │  2,097,664 │
│ (Conv2DTranspose)               │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ leaky_re_lu_5 (LeakyReLU)       │ (None, 64, 64, 512)       │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_3 (Conv2D)               │ (None, 64, 64, 3)         │     38,403 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 3,979,651 (15.18 MB)
 Trainable params: 3,979,651 (15.18 MB)
 Non-trainable params: 0 (0.00 B)
```

{{% /details %}}

## `train_step` 오버라이드 {#override-train_step}

```python
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(1337)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # 잠재 공간에서 무작위 지점을 샘플링합니다.
        batch_size = ops.shape(real_images)[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # 가짜 이미지로 디코딩
        generated_images = self.generator(random_latent_vectors)

        # 실제 이미지와 결합
        combined_images = ops.concatenate([generated_images, real_images], axis=0)

        # 진짜 이미지와 가짜 이미지를 구별하는 라벨을 조립합니다.
        labels = ops.concatenate(
            [ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0
        )
        # 라벨에 무작위 노이즈를 추가합니다. 중요한 기술입니다!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # 판별자를 트레이닝합니다.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # 잠재 공간에서 무작위 지점을 샘플링합니다.
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # "all real images"라고 적힌 라벨을 조립하세요.
        misleading_labels = ops.zeros((batch_size, 1))

        # 생성자를 트레이닝합니다. (판별자의 가중치는 업데이트해서는 *안 됩니다*!)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # 메트릭 업데이트
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
```

## 생성된 이미지를 주기적으로 저장하는 콜백을 만들기 {#create-a-callback-that-periodically-saves-generated-images}

```python
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator(42)

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = keras.random.normal(
            shape=(self.num_img, self.latent_dim), seed=self.seed_generator
        )
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save("generated_img_%03d_%d.png" % (epoch, i))
```

## 엔드투엔드 모델 트레이닝 {#train-the-end-to-end-model}

```python
epochs = 1  # 실제로는 ~100 에포크를 사용합니다.

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
    2/6332 [37m━━━━━━━━━━━━━━━━━━━━  9:54 94ms/step - d_loss: 0.6792 - g_loss: 0.7880

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1704214667.959762    1319 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.

 6332/6332 ━━━━━━━━━━━━━━━━━━━━ 557s 84ms/step - d_loss: 0.5616 - g_loss: 1.4099

<keras.src.callbacks.history.History at 0x7f251d32bc40>

```

{{% /details %}}

에포크 30 정도에 생성된 마지막 이미지 중 일부(그 이후로 결과가 계속 향상됩니다.):

![results](/images/examples/generative/dcgan_overriding_train_step/h5MtQZ7l.jpeg)
