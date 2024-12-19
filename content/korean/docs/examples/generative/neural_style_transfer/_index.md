---
title: 신경 스타일 전송
linkTitle: 신경 스타일 전송
toc: true
weight: 18
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2016/01/11  
**{{< t f_last_modified >}}** 2020/05/02  
**{{< t f_description >}}** 그래디언트 하강법을 사용하여, 참조 이미지의 스타일을 타겟 이미지로 전송하기

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/neural_style_transfer.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/neural_style_transfer.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

스타일 전송은 베이스 이미지와 동일한 "콘텐츠"를 가진 이미지를 생성하되,
"스타일"은 다른 이미지(일반적으로 예술적 이미지)의 스타일을 따르는 방식입니다.
이는 3가지 구성 요소를 가진 손실 함수의 최적화를 통해 이루어집니다:
"스타일 손실", "콘텐츠 손실", 그리고 "총 변동 손실(total variation loss)"

- 총 변동 손실은
  - 조합 이미지의 픽셀 간 지역적 공간 연속성을 부여하여, 시각적 일관성을 제공합니다.
- 스타일 손실은
  - 딥러닝이 관여하는 부분으로, 이는 심층 컨볼루션 신경망을 사용하여 정의됩니다.
    - 정확히 말하면, 스타일 손실은 ImageNet에 대해 트레이닝된 convnet의 서로 다른 레이어에서 추출된,
    - 베이스 이미지와 스타일 참조 이미지의 표현들에 대한 Gram matrices 간의 L2 거리의 합으로 구성됩니다.
  - 일반적인 아이디어는 서로 다른 공간적 규모에서 색상 및 텍스처 정보를 캡처하는 것입니다.
    - 이때 공간적 규모는 레이어의 깊이에 따라 정의되며, 상당히 큰 규모에서 처리됩니다.
- 콘텐츠 손실은
  - 기본 이미지(딥 레이어에서 추출된)의 특징과 조합 이미지의 특징 간의 L2 거리로 정의되며, 생성된 이미지가 원본과 충분히 유사하도록 유지합니다.

**참조:** [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)

## 셋업 {#setup}

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf
from keras.applications import vgg19

base_image_path = keras.utils.get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
style_reference_image_path = keras.utils.get_file(
    "starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg"
)
result_prefix = "paris_generated"

# 각 손실 요소들의 가중치
total_variation_weight = 1e-6
style_weight = 1e-6
content_weight = 2.5e-8

# 생성된 이미지의 크기.
width, height = keras.utils.load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Downloading data from https://i.imgur.com/F28w3Ac.jpg
 102437/102437 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Downloading data from https://i.imgur.com/9ooB60I.jpg
 935806/935806 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
```

{{% /details %}}

## 베이스(콘텐츠) 이미지와 스타일 참조 이미지를 살펴봅시다 {#lets-take-a-look-at-our-base-content-image-and-our-style-reference-image}

```python
from IPython.display import Image, display

display(Image(base_image_path))
display(Image(style_reference_image_path))
```

![jpeg](/images/examples/generative/neural_style_transfer/neural_style_transfer_5_0.jpg)

![jpeg](/images/examples/generative/neural_style_transfer/neural_style_transfer_5_1.jpg)

## 이미지 전처리 / 후처리 유틸리티 {#image-preprocessing-deprocessing-utilities}

```python
def preprocess_image(image_path):
    # 이미지를 열고, 크기를 조정하고, 적절한 텐서로 포맷(형식 변환)하는 유틸리티 함수
    img = keras.utils.load_img(image_path, target_size=(img_nrows, img_ncols))
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x):
    # 텐서를 유효한 이미지로 변환하는 유틸리티 함수
    x = x.reshape((img_nrows, img_ncols, 3))
    # 평균 픽셀값으로부터 영점 중심(zero-center) 제거
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x
```

## 스타일 변환 손실 계산 {#compute-the-style-transfer-loss}

먼저, 4개의 유틸리티 함수들을 정의해야 합니다:

- `gram_matrix`
  - 스타일 손실을 계산하는 데 사용
- `style_loss` 함수
  - 스타일 참조 이미지의 로컬 텍스처에 생성된 이미지가 가깝도록 유지
- `content_loss` 함수
  - 생성된 이미지의 높은 레벨 표현이 베이스 이미지와 가깝도록 유지
- `total_variation_loss` 함수
  - 생성된 이미지를 로컬에서 일관성(locally-coherent) 있게 유지하는 정규화 손실

```python
# 이미지 텐서의 gram matrix (특성 간 외적(outer product))


def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


# "style loss"는 참조 이미지의 스타일을 생성된 이미지에서 유지하기 위해 설계되었습니다.
# 이는 스타일 참조 이미지와 생성된 이미지의
# 특성 맵에서 gram matrices(스타일을 포착하는)을 기반으로 합니다.


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))


# 생성된 이미지에 베이스 이미지의 "콘텐츠"를 유지하도록 설계된 보조 손실 함수


def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))


# 세 번째 손실 함수, 총 변동 손실(total variation loss),
# 생성된 이미지가 로컬에서 일관성을 유지하도록 설계되었습니다.


def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))
```

다음으로, VGG19의 중간 활성값을 사전(dict) 형식으로 가져오는 특성 추출 모델을 생성해보겠습니다.

```python
# 사전 트레이닝된 ImageNet 가중치로 로드된 VGG19 모델을 빌드합니다.
model = vgg19.VGG19(weights="imagenet", include_top=False)

# 각 "key" 레이어의 상징적인 출력값을 가져옵니다. (고유한 이름을 부여했습니다)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# VGG19의 모든 레이어에 대해 활성화 값을 반환하는 모델을 설정합니다. (dict로)
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
 80134624/80134624 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step
```

{{% /details %}}

마지막으로, 스타일 전이 손실을 계산하는 코드입니다.

```python
# 스타일 손실에 사용할 레이어 목록
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# 콘텐츠 손실에 사용할 레이어
content_layer_name = "block5_conv2"


def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # 손실 초기화
    loss = tf.zeros(shape=())

    # 콘텐츠 손실 추가
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )

    # 스타일 손실 추가
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # 전체 변화 손실 추가
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss
```

## `tf.function` 데코레이터 추가하여 손실 및 그라디언트 계산 {#add-a-tffunction-decorator-to-loss-gradient-computation}

컴파일하여, 빠르게 만들기 위해, `tf.function` 데코레이터를 추가합니다.

```python
@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads
```

## 트레이닝 루프 {#the-training-loop}

손실을 최소화하기 위해 기본적인 그라디언트 하강법 스텝을 반복하여 실행하며, 100번 반복할 때마다 결과 이미지를 저장합니다.

학습률은 100 스텝마다 0.96으로 감소시킵니다.

```python
optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 4000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        img = deprocess_image(combination_image.numpy())
        fname = result_prefix + "_at_iteration_%d.png" % i
        keras.utils.save_img(fname, img)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Iteration 100: loss=11021.63
Iteration 200: loss=8516.82
Iteration 300: loss=7572.36
Iteration 400: loss=7062.23
Iteration 500: loss=6733.57
Iteration 600: loss=6498.27
Iteration 700: loss=6319.11
Iteration 800: loss=6176.94
Iteration 900: loss=6060.49
Iteration 1000: loss=5963.24
Iteration 1100: loss=5880.51
Iteration 1200: loss=5809.23
Iteration 1300: loss=5747.35
Iteration 1400: loss=5692.95
Iteration 1500: loss=5644.84
Iteration 1600: loss=5601.82
Iteration 1700: loss=5563.18
Iteration 1800: loss=5528.38
Iteration 1900: loss=5496.89
Iteration 2000: loss=5468.20
Iteration 2100: loss=5441.97
Iteration 2200: loss=5418.02
Iteration 2300: loss=5396.11
Iteration 2400: loss=5376.00
Iteration 2500: loss=5357.49
Iteration 2600: loss=5340.36
Iteration 2700: loss=5324.49
Iteration 2800: loss=5309.77
Iteration 2900: loss=5296.08
Iteration 3000: loss=5283.33
Iteration 3100: loss=5271.47
Iteration 3200: loss=5260.39
Iteration 3300: loss=5250.02
Iteration 3400: loss=5240.29
Iteration 3500: loss=5231.18
Iteration 3600: loss=5222.65
Iteration 3700: loss=5214.61
Iteration 3800: loss=5207.08
Iteration 3900: loss=5199.98
Iteration 4000: loss=5193.27
```

{{% /details %}}

4000번의 반복 후, 다음과 같은 결과를 얻게 됩니다:

```python
display(Image(result_prefix + "_at_iteration_4000.png"))
```

![png](/images/examples/generative/neural_style_transfer/neural_style_transfer_19_0.png)
