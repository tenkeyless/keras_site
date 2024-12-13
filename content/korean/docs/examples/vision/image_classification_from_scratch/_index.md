---
title: 처음부터 이미지 분류
linkTitle: 처음부터 이미지 분류
toc: true
weight: 1
type: docs
---

{{< keras/original checkedAt="2024-11-19" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2020/04/27  
**{{< t f_last_modified >}}** 2023/11/09  
**{{< t f_description >}}** Kaggle Cats vs Dogs 데이터세트를 사용하여 이미지 분류기를 처음부터 트레이닝합니다.

{{< keras/version v=3 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_from_scratch.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_from_scratch.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

이 예는 사전 트레이닝된 가중치나 미리 만들어진 Keras 애플리케이션 모델을 활용하지 않고,
디스크의 JPEG 이미지 파일에서 시작하여, 처음부터 이미지 분류를 수행하는 방법을 보여줍니다.
Kaggle Cats vs Dogs 이진 분류 데이터 세트에 대한 워크플로우를 보여드립니다.

`image_dataset_from_directory` 유틸리티를 사용해 데이터 세트를 생성하고,
이미지 표준화 및 데이터 보강을 위해 Keras 이미지 전처리 레이어를 사용합니다.

## 셋업 {#setup}

```python
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
```

## 데이터 로드: Cats vs Dogs 데이터 세트 {#load-the-data-the-cats-vs-dogs-dataset}

### Raw 데이터 다운로드 {#raw-data-download}

먼저, raw 데이터의 786M ZIP 아카이브를 다운로드해 보겠습니다:

```python
!curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
```

```python
!unzip -q kagglecatsanddogs_5340.zip
!ls
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  786M  100  786M    0     0  11.1M      0  0:01:10  0:01:10 --:--:-- 11.8M

 CDLA-Permissive-2.0.pdf           kagglecatsanddogs_5340.zip
 PetImages                'readme[1].txt'
 image_classification_from_scratch.ipynb
```

{{% /details %}}

이제 `Cat`과 `Dog`라는 두 개의 하위 폴더가 있는 `PetImages` 폴더가 생겼습니다.
각 하위 폴더에는 각 카테고리에 대한 이미지 파일이 들어 있습니다.

```python
!ls PetImages
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Cat  Dog
```

{{% /details %}}

### 손상된 이미지 필터링 {#filter-out-corrupted-images}

많은 실제 이미지 데이터로 작업할 때, 손상된 이미지가 흔히 발생합니다.
헤더에 "JFIF" 문자열이 없는 잘못 인코딩된 이미지를 필터링해 보겠습니다.

```python
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # 손상된 이미지 삭제
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Deleted 1590 images.
```

{{% /details %}}

## `Dataset` 생성 {#generate-a-dataset}

```python
image_size = (180, 180)
batch_size = 128

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Found 23410 files belonging to 2 classes.
Using 18728 files for training.
Using 4682 files for validation.
```

{{% /details %}}

## 데이터 시각화 {#visualize-the-data}

다음은 트레이닝 데이터 세트의 처음 9개 이미지입니다.

```python
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
```

![png](/images/examples/vision/image_classification_from_scratch/image_classification_from_scratch_14_1.png)

## 이미지 데이터 보강 사용 {#using-image-data-augmentation}

이미지 데이터 세트가 크지 않은 경우,
무작위 수평 뒤집기나 작은 무작위 회전 등 트레이닝 이미지에 무작위적이지만
사실적인 변형을 적용하여 샘플 다양성을 인위적으로 도입하는 것이 좋습니다.
이렇게 하면 모델이 트레이닝 데이터의 다양한 측면에 노출되는 동시에,
과적합 속도를 늦추는 데 도움이 됩니다.

```python
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images
```

데이터 세트의 처음 몇 개의 이미지에 `data_augmentation`을 반복적으로 적용하여,
보강된 샘플이 어떻게 보이는지 시각화해 보겠습니다:

```python
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")
```

![png](/images/examples/vision/image_classification_from_scratch/image_classification_from_scratch_18_1.png)

## 데이터 표준화 {#standardizing-the-data}

이미지는 데이터 세트에 의해 연속적인 `float32` 배치로 산출되기 때문에,
이미 표준 크기(180x180)로 되어 있습니다.
그러나, RGB 채널 값은 `[0, 255]` 범위에 있습니다.
이는 신경망에 이상적이지 않으며, 일반적으로 입력 값을 작게 만드는 것이 좋습니다.
여기서는 모델을 시작할 때 `Rescaling` 레이어를 사용하여,
값을 `[0, 1]` 범위로 표준화하겠습니다.

## 데이터 전처리를 위한 두 가지 옵션 {#two-options-to-preprocess-the-data}

`data_augmentation` 전처리기를 사용하는 방법에는 두 가지가 있습니다:

**옵션 1: 모델의 일부로 만들기**. 다음과 같습니다.:

```python
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
...  # 나머지 모델
```

이 옵션을 사용하면, 데이터 보강이 나머지 모델 실행과 동시에 _장치에서_ 발생하므로,
GPU 가속의 이점을 누릴 수 있습니다.

데이터 보강은 테스트 시에는 비활성 상태이므로,
입력 샘플은 `evaluate()` 또는 `predict()`을 호출할 때가 아니라,
`fit()` 중에만 보강된다는 점에 유의하세요.

GPU에서 트레이닝하는 경우, 이 옵션이 좋은 옵션일 수 있습니다.

**옵션 2: 데이터 세트에 적용**. 다음과 같이 보강된 이미지의 배치를 생성하는 데이터 세트를 얻습니다:

```python
augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))
```

이 옵션을 사용하면, 데이터 보강이 비동기적으로 **CPU**에서 이루어지며,
모델에 들어가기 전에 버퍼링됩니다.

CPU에서 트레이닝하는 경우,
이 옵션이 데이터 보강을 비동기식(asynchronous)으로 비차단적(non-blocking)으로 수행하므로,
더 나은 옵션입니다.

저희의 경우 두 번째 옵션을 사용하겠습니다.
어떤 것을 선택해야 할지 잘 모르겠다면,
두 번째 옵션(비동기 전처리)이 항상 확실한 선택입니다.

## 성능을 위한 데이터 세트 구성 {#configure-the-dataset-for-performance}

트레이닝 데이터 세트에 데이터 보강을 적용하고,
버퍼링된 프리페칭(prefetching)을 사용하여 I/O가 차단(blocking)되지 않고,
디스크에서 데이터를 가져올 수 있도록 해 보겠습니다:

```python
# 트레이닝 이미지에 `data_augmentation`을 적용
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# GPU 메모리에 샘플을 프리페칭(Prefetching)하면, GPU 활용도를 극대화할 수 있음.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
```

## 모델 빌드 {#build-a-model}

Xception 네트워크의 작은 버전을 구축하겠습니다. 아키텍처 최적화를 특별히 시도하지 않았으므로,
최적의 모델 구성을 체계적으로 검색하려면
[KerasTuner](https://github.com/keras-team/keras-tuner)를 사용하는 것이 좋습니다.

참고하세요:

- `data_augmentation` 전처리기로 모델을 시작한 다음, `Rescaling` 레이어를 추가합니다.
- 최종 분류 레이어 앞에 `Dropout` 레이어를 포함합니다.

```python
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # 엔트리 블록
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # 옆의 Residual을 설정

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Residual을 프로젝션
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Residual을 다시 더함
        previous_block_activation = x  # 다음 번의 옆의 Residual을 설정

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # logits를 반환하도록, `activation=None`으로 지정합니다.
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)
```

![png](/images/examples/vision/image_classification_from_scratch/image_classification_from_scratch_24_0.png)

## 모델 트레이닝 {#train-the-model}

```python
epochs = 25

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/25
...
Epoch 25/25
 147/147 ━━━━━━━━━━━━━━━━━━━━ 53s 354ms/step - acc: 0.9638 - loss: 0.0903 - val_acc: 0.9382 - val_loss: 0.1542

<keras.src.callbacks.history.History at 0x7f41003c24a0>
```

{{% /details %}}

전체 데이터 세트에서 25 에포크에 대해 트레이닝한 후, 90% 이상의 검증 정확도를 달성했습니다.
(실제로는 검증 성능이 저하되기 전에 50개 이상의 에포크에 대해 트레이닝할 수 있습니다)

## 새 데이터에 대한 추론 실행 {#run-inference-on-new-data}

추론 시에는 데이터 보강 및 드롭아웃이 비활성 상태입니다.

```python
img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # 배치 축 생성

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
 1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step
This image is 94.30% cat and 5.70% dog.
```

{{% /details %}}

![png](/images/examples/vision/image_classification_from_scratch/image_classification_from_scratch_29_1.png)
