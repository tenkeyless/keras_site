---
title: PixelCNN
linkTitle: PixelCNN
toc: true
weight: 15
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** [ADMoreau](https://github.com/ADMoreau)  
**{{< t f_date_created >}}** 2020/05/17  
**{{< t f_last_modified >}}** 2020/05/23  
**{{< t f_description >}}** Keras로 구현한 PixelCNN

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/pixelcnn.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/pixelcnn.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

PixelCNN은 2016년 van den Oord et al.에 의해 제안된 생성 모델입니다.
(참조: [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328))
이 모델은 이미지나 다른 데이터 타입을 입력 벡터로부터 반복적으로 생성하는 데 사용됩니다.
이전 요소들의 확률 분포가 이후 요소들의 확률 분포를 결정하는 방식으로 동작합니다.
아래 예시에서는, 이미지를 픽셀별로 이러한 방식으로 생성합니다.
마스크된 컨볼루션 커널을 통해 이전에 생성된 픽셀(좌상단 기점)의 데이터만을 보고 이후 픽셀을 생성합니다.
추론 과정에서는, 네트워크의 출력이 확률 분포로 사용되며,
여기서 새로운 픽셀 값이 샘플링되어 새로운 이미지가 생성됩니다.
(이 예시에서는 MNIST 데이터셋을 사용하며, 픽셀 값은 흑백입니다)

```python
import numpy as np
import keras
from keras import layers
from keras import ops
from tqdm import tqdm
```

## 데이터 얻기 {#getting-the-data}

```python
# 모델 및 데이터 파라미터
num_classes = 10
input_shape = (28, 28, 1)
n_residual_blocks = 5
# 트레이닝 및 테스트 데이터 분할
(x, _), (y, _) = keras.datasets.mnist.load_data()
# 모든 이미지를 하나로 연결(concatenate)
data = np.concatenate((x, y), axis=0)
# 최대 256 값의 33% 이하인 모든 픽셀 값을 0으로 반올림.
# 이 값보다 높은 모든 값은 1로 반올림하여, 모든 값을 0 또는 1로 만듭니다.
data = np.where(data < (0.33 * 256), 0, 1)
data = data.astype(np.float32)
```

## 모델을 위한 필수 레이어 두 가지 클래스를 생성 {#create-two-classes-for-the-requisite-layers-for-the-model}

```python
# 첫 번째 레이어는 PixelCNN 레이어입니다.
# 이 레이어는 2D 컨볼루션 레이어를 기반으로 하지만, 마스킹을 포함합니다.
class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super().__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Conv2D 레이어를 빌드하여 커널 변수를 초기화합니다.
        self.conv.build(input_shape)
        # 초기화된 커널을 사용하여 마스크를 생성합니다.
        kernel_shape = ops.shape(self.conv.kernel)
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# 다음으로, Residual 블록 레이어를 빌드합니다.
# 이것은 PixelConvLayer를 기반으로 하는 일반적인 Residual 블록입니다.
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])
```

## 원본 논문 기반 모델 빌드 {#build-the-model-based-on-the-original-paper}

```python
inputs = keras.Input(shape=input_shape, batch_size=128)
x = PixelConvLayer(
    mask_type="A", filters=128, kernel_size=7, activation="relu", padding="same"
)(inputs)

for _ in range(n_residual_blocks):
    x = ResidualBlock(filters=128)(x)

for _ in range(2):
    x = PixelConvLayer(
        mask_type="B",
        filters=128,
        kernel_size=1,
        strides=1,
        activation="relu",
        padding="valid",
    )(x)

out = keras.layers.Conv2D(
    filters=1, kernel_size=1, strides=1, activation="sigmoid", padding="valid"
)(x)

pixel_cnn = keras.Model(inputs, out)
adam = keras.optimizers.Adam(learning_rate=0.0005)
pixel_cnn.compile(optimizer=adam, loss="binary_crossentropy")

pixel_cnn.summary()
pixel_cnn.fit(
    x=data, y=data, batch_size=128, epochs=50, validation_split=0.1, verbose=2
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Model: "functional_1"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (128, 28, 28, 1)          │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ pixel_conv_layer                │ (128, 28, 28, 128)        │      6,400 │
│ (PixelConvLayer)                │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ residual_block (ResidualBlock)  │ (128, 28, 28, 128)        │     98,624 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ residual_block_1                │ (128, 28, 28, 128)        │     98,624 │
│ (ResidualBlock)                 │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ residual_block_2                │ (128, 28, 28, 128)        │     98,624 │
│ (ResidualBlock)                 │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ residual_block_3                │ (128, 28, 28, 128)        │     98,624 │
│ (ResidualBlock)                 │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ residual_block_4                │ (128, 28, 28, 128)        │     98,624 │
│ (ResidualBlock)                 │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ pixel_conv_layer_6              │ (128, 28, 28, 128)        │     16,512 │
│ (PixelConvLayer)                │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ pixel_conv_layer_7              │ (128, 28, 28, 128)        │     16,512 │
│ (PixelConvLayer)                │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_18 (Conv2D)              │ (128, 28, 28, 1)          │        129 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 532,673 (2.03 MB)
 Trainable params: 532,673 (2.03 MB)
 Non-trainable params: 0 (0.00 B)
```

```plain
Epoch 1/50
493/493 - 26s - 53ms/step - loss: 0.1137 - val_loss: 0.0933
Epoch 2/50
493/493 - 14s - 29ms/step - loss: 0.0915 - val_loss: 0.0901
Epoch 3/50
493/493 - 14s - 29ms/step - loss: 0.0893 - val_loss: 0.0888
Epoch 4/50
493/493 - 14s - 29ms/step - loss: 0.0882 - val_loss: 0.0880
Epoch 5/50
493/493 - 14s - 29ms/step - loss: 0.0874 - val_loss: 0.0870
Epoch 6/50
493/493 - 14s - 29ms/step - loss: 0.0867 - val_loss: 0.0867
Epoch 7/50
493/493 - 14s - 29ms/step - loss: 0.0863 - val_loss: 0.0867
Epoch 8/50
493/493 - 14s - 29ms/step - loss: 0.0859 - val_loss: 0.0860
Epoch 9/50
493/493 - 14s - 29ms/step - loss: 0.0855 - val_loss: 0.0856
Epoch 10/50
493/493 - 14s - 29ms/step - loss: 0.0853 - val_loss: 0.0861
Epoch 11/50
493/493 - 14s - 29ms/step - loss: 0.0850 - val_loss: 0.0860
Epoch 12/50
493/493 - 14s - 29ms/step - loss: 0.0847 - val_loss: 0.0873
Epoch 13/50
493/493 - 14s - 29ms/step - loss: 0.0846 - val_loss: 0.0852
Epoch 14/50
493/493 - 14s - 29ms/step - loss: 0.0844 - val_loss: 0.0846
Epoch 15/50
493/493 - 14s - 29ms/step - loss: 0.0842 - val_loss: 0.0848
Epoch 16/50
493/493 - 14s - 29ms/step - loss: 0.0840 - val_loss: 0.0843
Epoch 17/50
493/493 - 14s - 29ms/step - loss: 0.0838 - val_loss: 0.0847
Epoch 18/50
493/493 - 14s - 29ms/step - loss: 0.0837 - val_loss: 0.0841
Epoch 19/50
493/493 - 14s - 29ms/step - loss: 0.0835 - val_loss: 0.0842
Epoch 20/50
493/493 - 14s - 29ms/step - loss: 0.0834 - val_loss: 0.0844
Epoch 21/50
493/493 - 14s - 29ms/step - loss: 0.0834 - val_loss: 0.0843
Epoch 22/50
493/493 - 14s - 29ms/step - loss: 0.0832 - val_loss: 0.0838
Epoch 23/50
493/493 - 14s - 29ms/step - loss: 0.0831 - val_loss: 0.0840
Epoch 24/50
493/493 - 14s - 29ms/step - loss: 0.0830 - val_loss: 0.0841
Epoch 25/50
493/493 - 14s - 29ms/step - loss: 0.0829 - val_loss: 0.0837
Epoch 26/50
493/493 - 14s - 29ms/step - loss: 0.0828 - val_loss: 0.0837
Epoch 27/50
493/493 - 14s - 29ms/step - loss: 0.0827 - val_loss: 0.0836
Epoch 28/50
493/493 - 14s - 29ms/step - loss: 0.0827 - val_loss: 0.0836
Epoch 29/50
493/493 - 14s - 29ms/step - loss: 0.0825 - val_loss: 0.0838
Epoch 30/50
493/493 - 14s - 29ms/step - loss: 0.0825 - val_loss: 0.0834
Epoch 31/50
493/493 - 14s - 29ms/step - loss: 0.0824 - val_loss: 0.0832
Epoch 32/50
493/493 - 14s - 29ms/step - loss: 0.0823 - val_loss: 0.0833
Epoch 33/50
493/493 - 14s - 29ms/step - loss: 0.0822 - val_loss: 0.0836
Epoch 34/50
493/493 - 14s - 29ms/step - loss: 0.0822 - val_loss: 0.0832
Epoch 35/50
493/493 - 14s - 29ms/step - loss: 0.0821 - val_loss: 0.0832
Epoch 36/50
493/493 - 14s - 29ms/step - loss: 0.0820 - val_loss: 0.0835
Epoch 37/50
493/493 - 14s - 29ms/step - loss: 0.0820 - val_loss: 0.0834
Epoch 38/50
493/493 - 14s - 29ms/step - loss: 0.0819 - val_loss: 0.0833
Epoch 39/50
493/493 - 14s - 29ms/step - loss: 0.0818 - val_loss: 0.0832
Epoch 40/50
493/493 - 14s - 29ms/step - loss: 0.0818 - val_loss: 0.0834
Epoch 41/50
493/493 - 14s - 29ms/step - loss: 0.0817 - val_loss: 0.0832
Epoch 42/50
493/493 - 14s - 29ms/step - loss: 0.0816 - val_loss: 0.0834
Epoch 43/50
493/493 - 14s - 29ms/step - loss: 0.0816 - val_loss: 0.0839
Epoch 44/50
493/493 - 14s - 29ms/step - loss: 0.0815 - val_loss: 0.0831
Epoch 45/50
493/493 - 14s - 29ms/step - loss: 0.0815 - val_loss: 0.0832
Epoch 46/50
493/493 - 14s - 29ms/step - loss: 0.0814 - val_loss: 0.0835
Epoch 47/50
493/493 - 14s - 29ms/step - loss: 0.0814 - val_loss: 0.0830
Epoch 48/50
493/493 - 14s - 29ms/step - loss: 0.0813 - val_loss: 0.0832
Epoch 49/50
493/493 - 14s - 29ms/step - loss: 0.0812 - val_loss: 0.0833
Epoch 50/50
493/493 - 14s - 29ms/step - loss: 0.0812 - val_loss: 0.0831

<keras.src.callbacks.history.History at 0x7f45e6d78760>
```

{{% /details %}}

## 데모 {#demonstration}

PixelCNN은 전체 이미지를 한 번에 생성할 수 없습니다.
대신 각 픽셀을 순차적으로 생성하며,
마지막으로 생성된 픽셀을 현재 이미지에 추가한 후 다시 모델에 피드하여 이 과정을 반복해야 합니다.

```python
from IPython.display import Image, display

# 빈 픽셀 배열을 생성합니다.
batch = 4
pixels = np.zeros(shape=(batch,) + (pixel_cnn.input_shape)[1:])
batch, rows, cols, channels = pixels.shape

# 픽셀을 순차적으로 생성해야 하므로, 픽셀을 반복합니다.
for row in tqdm(range(rows)):
    for col in range(cols):
        for channel in range(channels):
            # 전체 배열을 모델에 피드하고, 다음 픽셀의 값 확률을 가져옵니다.
            probs = pixel_cnn.predict(pixels)[:, row, col, channel]
            # 확률을 사용하여 픽셀 값을 선택하고 이미지 프레임에 값을 추가합니다.
            pixels[:, row, col, channel] = ops.ceil(
                probs - keras.random.uniform(probs.shape)
            )


def deprocess_image(x):
    # 단일 채널 흑백 이미지를 RGB 값으로 쌓습니다.
    x = np.stack((x, x, x), 2)
    # 전처리 해제
    x *= 255.0
    # uint8로 변환하고 [0, 255]의 유효 범위로 클리핑합니다.
    x = np.clip(x, 0, 255).astype("uint8")
    return x


# 생성된 이미지를 반복하면서 matplotlib을 사용해 이미지를 플롯합니다.
for i, pic in enumerate(pixels):
    keras.utils.save_img(
        "generated_image_{}.png".format(i), deprocess_image(np.squeeze(pic, -1))
    )

display(Image("generated_image_0.png"))
display(Image("generated_image_1.png"))
display(Image("generated_image_2.png"))
display(Image("generated_image_3.png"))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
100%|███████████████████████████████████████████████████████████████████████████| 28/28 [00:06<00:00,  4.51it/s]
```

{{% /details %}}

![png](/images/examples/generative/pixelcnn/pixelcnn_10_57.png)

![png](/images/examples/generative/pixelcnn/pixelcnn_10_58.png)

![png](/images/examples/generative/pixelcnn/pixelcnn_10_59.png)

![png](/images/examples/generative/pixelcnn/pixelcnn_10_60.png)
