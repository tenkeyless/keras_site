---
title: Deep Dream
linkTitle: Deep Dream
toc: true
weight: 13
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2016/01/13  
**{{< t f_last_modified >}}** 2020/05/02  
**{{< t f_description >}}** Keras로 Deep Dream 만들기

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/deep_dream.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/deep_dream.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

"Deep dream"은 이미지 분류 모델을 사용하여,
입력 이미지에서 특정 레이어(때로는 특정 레이어의 특정 유닛)의 활성화를 최대화하기 위해,
그래디언트 상승(gradient ascent)을 실행하는 이미지 필터링 기법입니다.
이 과정은 환각 같은 시각적 효과를 만들어냅니다.

이 기술은 2015년 7월, Google의 Alexander Mordvintsev에 의해 처음 도입되었습니다.

과정:

- 원본 이미지를 로드합니다.
- 가장 작은 스케일에서부터 가장 큰 스케일까지의, 여러 처리 스케일("옥타브, octaves")을 정의합니다.
- 원본 이미지를 가장 작은 스케일로 크기를 조정합니다.
- 각 스케일에 대해, 가장 작은 스케일(즉, 현재 스케일)에서 시작하여 다음을 수행합니다:
  - 그래디언트 상승(gradient ascent) 실행
  - 이미지를 다음 스케일로 업스케일
  - 업스케일 시 손실된, 디테일 재주입
- 원래 크기로 돌아오면 중지합니다.
  업스케일 과정에서 손실된 디테일을 얻기 위해, 원본 이미지를 축소하고,
  다시 업스케일한 다음, 결과를 (리사이즈된) 원본 이미지와 비교합니다.

## 셋업 {#setup}

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras.applications import inception_v3

base_image_path = keras.utils.get_file("sky.jpg", "https://i.imgur.com/aGBdQyK.jpg")
result_prefix = "sky_dream"

# 우리가 활성화를 최대화하려는 레이어의 이름과 최종 손실에서의 가중치입니다.
# 새로운 시각적 효과를 얻기 위해 이러한 설정을 조정할 수 있습니다.
layer_settings = {
    "mixed4": 1.0,
    "mixed5": 1.5,
    "mixed6": 2.0,
    "mixed7": 2.5,
}

# 이러한 하이퍼파라미터를 조정하여, 새로운 효과를 얻을 수도 있습니다.
step = 0.01  # 그래디언트 상승 스텝 크기
num_octave = 3  # 그래디언트 상승을 실행할 스케일 수
octave_scale = 1.4  # 스케일 간의 크기 비율
iterations = 20  # 스케일당 상승 스텝 수
max_loss = 15.0
```

이것이 우리의 베이스 이미지입니다:

```python
from IPython.display import Image, display

display(Image(base_image_path))
```

![jpeg](/images/examples/generative/deep_dream/deep_dream_5_0.jpg)

이미지 전처리 및 후처리 유틸리티를 설정해봅시다:

```python
def preprocess_image(image_path):
    # 이미지를 열고, 크기를 조정하고, 적절한 배열로 포맷팅하는 유틸리티 함수
    img = keras.utils.load_img(image_path)
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # NumPy 배열을 유효한 이미지로 변환하는 유틸리티 함수
    x = x.reshape((x.shape[1], x.shape[2], 3))
    # Inception v3 전처리 되돌리기(undo)
    x /= 2.0
    x += 0.5
    x *= 255.0
    # uint8로 변환하고 유효 범위 [0, 255]로 클립
    x = np.clip(x, 0, 255).astype("uint8")
    return x
```

## Deep Dream 손실 계산 {#compute-the-deep-dream-loss}

먼저, 입력 이미지가 주어졌을 때 목표 레이어의 활성화를 가져오기 위해, 특성 추출 모델을 빌드합니다.

```python
# 사전 트레이닝된 ImageNet 가중치로 로드된 InceptionV3 모델을 빌드합니다.
model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

# 각 "주요" 레이어의 상징적(symbolic) 출력을 가져옵니다. (고유한 이름을 부여했습니다)
outputs_dict = dict(
    [
        (layer.name, layer.output)
        for layer in [model.get_layer(name) for name in layer_settings.keys()]
    ]
)

# 각 목표 레이어에 대한 활성화 값을 (딕셔너리 형태로) 반환하는 모델을 설정합니다.
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
```

실제 손실 계산은 매우 간단합니다:

```python
def compute_loss(input_image):
    features = feature_extractor(input_image)
    # 손실 초기화
    loss = tf.zeros(shape=())
    for name in features.keys():
        coeff = layer_settings[name]
        activation = features[name]
        # 손실 계산 시 경계 아티팩트를 피하기 위해 경계 픽셀은 제외합니다.
        scaling = tf.reduce_prod(tf.cast(tf.shape(activation), "float32"))
        loss += coeff * tf.reduce_sum(tf.square(activation[:, 2:-2, 2:-2, :])) / scaling
    return loss
```

## 하나의 옥타브에 대한 그래디언트 상승 루프 설정 {#set-up-the-gradient-ascent-loop-for-one-octave}

```python
@tf.function
def gradient_ascent_step(img, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img)
    # 그래디언트 계산
    grads = tape.gradient(loss, img)
    # 그래디언트 정규화
    grads /= tf.maximum(tf.reduce_mean(tf.abs(grads)), 1e-6)
    img += learning_rate * grads
    return loss, img


def gradient_ascent_loop(img, iterations, learning_rate, max_loss=None):
    for i in range(iterations):
        loss, img = gradient_ascent_step(img, learning_rate)
        if max_loss is not None and loss > max_loss:
            break
        print("... Loss value at step %d: %.2f" % (i, loss))
    return img
```

## 다양한 옥타브를 반복하면서 트레이닝 루프 실행 {#run-the-training-loop-iterating-over-different-octaves}

```python
original_img = preprocess_image(base_image_path)
original_shape = original_img.shape[1:3]

successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale**i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])

img = tf.identity(original_img)  # 복사본을 만듭니다.
for i, shape in enumerate(successive_shapes):
    print("Processing octave %d with shape %s" % (i, shape))
    img = tf.image.resize(img, shape)
    img = gradient_ascent_loop(
        img, iterations=iterations, learning_rate=step, max_loss=max_loss
    )
    upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
    same_size_original = tf.image.resize(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = tf.image.resize(original_img, shape)

keras.utils.save_img(result_prefix + ".png", deprocess_image(img.numpy()))
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Processing octave 0 with shape (326, 489)
... Loss value at step 0: 0.45
... Loss value at step 1: 0.63
... Loss value at step 2: 0.91
... Loss value at step 3: 1.24
... Loss value at step 4: 1.57
... Loss value at step 5: 1.91
... Loss value at step 6: 2.20
... Loss value at step 7: 2.50
... Loss value at step 8: 2.82
... Loss value at step 9: 3.11
... Loss value at step 10: 3.40
... Loss value at step 11: 3.70
... Loss value at step 12: 3.95
... Loss value at step 13: 4.20
... Loss value at step 14: 4.48
... Loss value at step 15: 4.72
... Loss value at step 16: 4.99
... Loss value at step 17: 5.23
... Loss value at step 18: 5.47
... Loss value at step 19: 5.69
Processing octave 1 with shape (457, 685)
... Loss value at step 0: 1.11
... Loss value at step 1: 1.77
... Loss value at step 2: 2.35
... Loss value at step 3: 2.82
... Loss value at step 4: 3.25
... Loss value at step 5: 3.67
... Loss value at step 6: 4.05
... Loss value at step 7: 4.44
... Loss value at step 8: 4.79
... Loss value at step 9: 5.15
... Loss value at step 10: 5.50
... Loss value at step 11: 5.84
... Loss value at step 12: 6.18
... Loss value at step 13: 6.49
... Loss value at step 14: 6.82
... Loss value at step 15: 7.12
... Loss value at step 16: 7.42
... Loss value at step 17: 7.71
... Loss value at step 18: 8.01
... Loss value at step 19: 8.30
Processing octave 2 with shape (640, 960)
... Loss value at step 0: 1.27
... Loss value at step 1: 2.02
... Loss value at step 2: 2.63
... Loss value at step 3: 3.15
... Loss value at step 4: 3.66
... Loss value at step 5: 4.12
... Loss value at step 6: 4.58
... Loss value at step 7: 5.01
... Loss value at step 8: 5.42
... Loss value at step 9: 5.80
... Loss value at step 10: 6.19
... Loss value at step 11: 6.54
... Loss value at step 12: 6.89
... Loss value at step 13: 7.22
... Loss value at step 14: 7.57
... Loss value at step 15: 7.88
... Loss value at step 16: 8.21
... Loss value at step 17: 8.53
... Loss value at step 18: 8.80
... Loss value at step 19: 9.10
```

{{% /details %}}

결과를 표시합니다.

[Hugging Face Hub](https://huggingface.co/keras-io/deep-dream)에 호스팅된 트레이닝된 모델을 사용해 볼 수 있으며,
[Hugging Face Spaces](https://huggingface.co/spaces/keras-io/deep-dream)에서 데모를 시도해 볼 수 있습니다.

```python
display(Image(result_prefix + ".png"))
```

![png](/images/examples/generative/deep_dream/deep_dream_17_0.png)
