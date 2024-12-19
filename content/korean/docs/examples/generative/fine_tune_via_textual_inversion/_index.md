---
title: Textual Inversion을 통해 StableDiffusion의 새로운 개념 가르치기
linkTitle: StableDiffusion의 새로운 개념
toc: true
weight: 5
type: docs
---

{{< keras/original checkedAt="2024-11-22" >}}

**{{< t f_author >}}** Ian Stenbit, [lukewood](https://lukewood.xyz)  
**{{< t f_date_created >}}** 2022/12/09  
**{{< t f_last_modified >}}** 2022/12/09  
**{{< t f_description >}}** KerasCV의 StableDiffusion 구현을 통해 새로운 시각적 개념을 알아봅니다.

{{< keras/version v=2 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/fine_tune_via_textual_inversion.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/fine_tune_via_textual_inversion.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## Textual Inversion {#textual-inversion}

출시 이후, StableDiffusion은 생성적 머신러닝 커뮤니티에서 빠르게 인기를 얻었습니다.
트래픽 양이 많아 오픈 소스 기여 개선, 신속한 엔지니어링, 심지어 새로운 알고리즘의 발명까지 이어졌습니다.

아마도 가장 인상적인 새로운 알고리즘은 [Textual Inversion](https://github.com/rinongal/textual_inversion)으로,
[_이미지는 한 단어의 가치가 있다: Textual Inversion을 사용하여 텍스트-이미지 생성 개인화_](https://textual-inversion.github.io/)에 제시되어 있습니다.

Textual Inversion은 미세 조정을 사용하여 이미지 생성기에 특정 시각적 개념을 가르치는 프로세스입니다.
아래 다이어그램에서, 저자가 모델에 새로운 개념을 가르치고, 이를 "S\_\*"라고 부르는, 이 프로세스의 예를 볼 수 있습니다.

![https://i.imgur.com/KqEeBsM.jpg](/images/examples/generative/fine_tune_via_textual_inversion/KqEeBsM.jpeg)

개념적으로, textual inversion은 새 텍스트 토큰에 대한 토큰 임베딩을 학습하여,
StableDiffusion의 나머지 구성 요소를 동결하는 방식으로 작동합니다.

이 가이드에서는 Textual-Inversion 알고리즘을 사용하여,
KerasCV에 제공된 StableDiffusion 모델을 미세 조정하는 방법을 보여줍니다.
가이드를 마치면 "Gandalf the Gray as a \<my-funny-cat-token\>"을 작성할 수 있게 됩니다.

![https://i.imgur.com/rcb1Yfx.png](/images/examples/generative/fine_tune_via_textual_inversion/rcb1Yfx.png)

먼저, 필요한 패키지를 import 하고,
StableDiffusion 인스턴스를 생성하여,
일부 하위 구성 요소를 사용하여 미세 조정을 수행해 보겠습니다.

```python
!pip install -q git+https://github.com/keras-team/keras-cv.git
!pip install -q tensorflow==2.11.0
```

```python
import math

import keras_cv
import numpy as np
import tensorflow as tf
from keras_cv import layers as cv_layers
from keras_cv.models.stable_diffusion import NoiseScheduler
from tensorflow import keras
import matplotlib.pyplot as plt

stable_diffusion = keras_cv.models.StableDiffusion()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
By using this model checkpoint, you acknowledge that its usage is subject to the terms of the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE
```

{{% /details %}}

다음으로, 생성된 이미지를 보여주기 위한 시각화 유틸리티를 정의해 보겠습니다.

```python
def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
```

## 텍스트-이미지 쌍 데이터 세트 모으기 {#assembling-a-text-image-pair-dataset}

새로운 토큰의 임베딩을 트레이닝하기 위해, 먼저 텍스트-이미지 쌍으로 구성된 데이터 세트를 조립해야 합니다.
데이터 세트의 각 샘플에는,
StableDiffusion에 가르치는 개념의 이미지와 이미지의 내용을 정확하게 나타내는 캡션이,
포함되어야 합니다.
이 튜토리얼에서는, StableDiffusion에 Luke와 Ian의 GitHub 아바타 개념을 가르칩니다.

![gh-avatars](/images/examples/generative/fine_tune_via_textual_inversion/WyEHDIR.jpeg)

먼저, 고양이 인형의 이미지 데이터 세트를 구성해 보겠습니다.

```python
def assemble_image_dataset(urls):
    # 모든 원격 파일 가져오기
    files = [tf.keras.utils.get_file(origin=url) for url in urls]

    # 이미지 크기 조절
    resize = keras.layers.Resizing(height=512, width=512, crop_to_aspect_ratio=True)
    images = [keras.utils.load_img(img) for img in files]
    images = [keras.utils.img_to_array(img) for img in images]
    images = np.array([resize(img) for img in images])

    # StableDiffusion 이미지 인코더는 이미지가 [-1, 1] 픽셀 값 범위로 정규화되어야 합니다.
    images = images / 127.5 - 1

    # tf.data.Dataset을 생성합니다
    image_dataset = tf.data.Dataset.from_tensor_slices(images)

    # 셔플하고 랜덤 노이즈를 도입합니다.
    image_dataset = image_dataset.shuffle(50, reshuffle_each_iteration=True)
    image_dataset = image_dataset.map(
        cv_layers.RandomCropAndResize(
            target_size=(512, 512),
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(1.0, 1.0),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    image_dataset = image_dataset.map(
        cv_layers.RandomFlip(mode="horizontal"),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return image_dataset
```

다음으로, 텍스트 데이터 세트를 조립합니다.

```python
MAX_PROMPT_LENGTH = 77
placeholder_token = "<my-funny-cat-token>"


def pad_embedding(embedding):
    return embedding + (
        [stable_diffusion.tokenizer.end_of_text] * (MAX_PROMPT_LENGTH - len(embedding))
    )


stable_diffusion.tokenizer.add_tokens(placeholder_token)


def assemble_text_dataset(prompts):
    prompts = [prompt.format(placeholder_token) for prompt in prompts]
    embeddings = [stable_diffusion.tokenizer.encode(prompt) for prompt in prompts]
    embeddings = [np.array(pad_embedding(embedding)) for embedding in embeddings]
    text_dataset = tf.data.Dataset.from_tensor_slices(embeddings)
    text_dataset = text_dataset.shuffle(100, reshuffle_each_iteration=True)
    return text_dataset
```

마지막으로, 우리는 데이터 세트를 zip하여, 텍스트-이미지 쌍 데이터 세트를 생성합니다.

```python
def assemble_dataset(urls, prompts):
    image_dataset = assemble_image_dataset(urls)
    text_dataset = assemble_text_dataset(prompts)
    # 이미지 데이터 세트는 매우 짧으므로,
    # 텍스트 프롬프트 데이터 세트의 길이에 맞게 반복합니다.
    image_dataset = image_dataset.repeat()
    # 우리는 텍스트 프롬프트 데이터 세트를 사용하여, 데이터 세트의 길이를 결정합니다.
    # 프롬프트가 비교적 적기 때문에, 데이터 세트를 5번 반복합니다.
    # 우리는 이것이 일화적으로 결과를 개선한다는 것을 발견했습니다.
    text_dataset = text_dataset.repeat(5)
    return tf.data.Dataset.zip((image_dataset, text_dataset))
```

프롬프트가 설명적(descriptive)이 되도록 하기 위해, 우리는 매우 일반적인(generic) 프롬프트를 사용합니다.

샘플 이미지와 프롬프트로 이것을 시도해 보겠습니다.

```python
train_ds = assemble_dataset(
    urls=[
        "https://i.imgur.com/VIedH1X.jpg",
        "https://i.imgur.com/eBw13hE.png",
        "https://i.imgur.com/oJ3rSg7.png",
        "https://i.imgur.com/5mCL6Df.jpg",
        "https://i.imgur.com/4Q6WWyI.jpg",
    ],
    prompts=[
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ],
)
```

## 프롬프트 정확성의 중요성에 대하여 {#on-the-importance-of-prompt-accuracy}

이 가이드를 처음 쓰려고 할 때, 우리는 데이터 세트에 이 고양이 인형 그룹의 이미지를 포함시켰지만,
위에 나열된 일반적인 프롬프트를 계속 사용했습니다.
결과는 일화적으로 좋지 않았습니다.
예를 들어, 이 방법을 사용한 고양이 인형 간달프는 다음과 같습니다.

![mediocre-wizard](/images/examples/generative/fine_tune_via_textual_inversion/Thq7XOu.jpeg)

개념적으로는 비슷하지만, 가능한 한 좋지 않습니다.

이를 해결하기 위해, 이미지를 개별 고양이 인형 이미지와 고양이 인형 그룹 이미지로 분할하는 실험을 시작했습니다.
이렇게 분할한 후, 그룹 사진에 대한 새로운 프롬프트를 생각해냈습니다.

내용을 정확하게 표현하는 텍스트-이미지 쌍에 대한 트레이닝은 결과의 품질을 _상당히_ 향상시켰습니다.
이는 프롬프트 정확성의 중요성을 말해줍니다.

이미지를 개별 이미지와 그룹 이미지로 분리하는 것 외에도,
("{}의 어두운 사진"과 같은) 일부 부정확한 프롬프트도 제거합니다.

이를 염두에 두고, 아래에 최종 트레이닝 데이터 세트를 조립합니다.

```python
single_ds = assemble_dataset(
    urls=[
        "https://i.imgur.com/VIedH1X.jpg",
        "https://i.imgur.com/eBw13hE.png",
        "https://i.imgur.com/oJ3rSg7.png",
        "https://i.imgur.com/5mCL6Df.jpg",
        "https://i.imgur.com/4Q6WWyI.jpg",
    ],
    prompts=[
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ],
)
```

![https://i.imgur.com/gQCRjK6.png](/images/examples/generative/fine_tune_via_textual_inversion/gQCRjK6.png)

훌륭해 보이네요!

다음으로, GitHub 아바타 그룹의 데이터 세트를 조립합니다.

```python
group_ds = assemble_dataset(
    urls=[
        "https://i.imgur.com/yVmZ2Qa.jpg",
        "https://i.imgur.com/JbyFbZJ.jpg",
        "https://i.imgur.com/CCubd3q.jpg",
    ],
    prompts=[
        "a photo of a group of {}",
        "a rendering of a group of {}",
        "a cropped photo of the group of {}",
        "the photo of a group of {}",
        "a photo of a clean group of {}",
        "a photo of my group of {}",
        "a photo of a cool group of {}",
        "a close-up photo of a group of {}",
        "a bright photo of the group of {}",
        "a cropped photo of a group of {}",
        "a photo of the group of {}",
        "a good photo of the group of {}",
        "a photo of one group of {}",
        "a close-up photo of the group of {}",
        "a rendition of the group of {}",
        "a photo of the clean group of {}",
        "a rendition of a group of {}",
        "a photo of a nice group of {}",
        "a good photo of a group of {}",
        "a photo of the nice group of {}",
        "a photo of the small group of {}",
        "a photo of the weird group of {}",
        "a photo of the large group of {}",
        "a photo of a cool group of {}",
        "a photo of a small group of {}",
    ],
)
```

![https://i.imgur.com/GY9Pf3D.png](/images/examples/generative/fine_tune_via_textual_inversion/GY9Pf3D.png)

마지막으로, 두 데이터 세트를 연결(concatenate)합니다.

```python
train_ds = single_ds.concatenate(group_ds)
train_ds = train_ds.batch(1).shuffle(
    train_ds.cardinality(), reshuffle_each_iteration=True
)
```

## 텍스트 인코더에 새 토큰 추가 {#adding-a-new-token-to-the-text-encoder}

다음으로, StableDiffusion 모델에 대한 새로운 텍스트 인코더를 만들고,
''에 대한 새로운 임베딩을 모델에 추가합니다.

```python
tokenized_initializer = stable_diffusion.tokenizer.encode("cat")[1]
new_weights = stable_diffusion.text_encoder.layers[2].token_embedding(
    tf.constant(tokenized_initializer)
)

# tokenizer 대신 .vocab의 len을 가져옵니다.
new_vocab_size = len(stable_diffusion.tokenizer.vocab)

# 임베딩 레이어는 텍스트 인코더의 2번째 레이어입니다.
old_token_weights = stable_diffusion.text_encoder.layers[
    2
].token_embedding.get_weights()
old_position_weights = stable_diffusion.text_encoder.layers[
    2
].position_embedding.get_weights()

old_token_weights = old_token_weights[0]
new_weights = np.expand_dims(new_weights, axis=0)
new_weights = np.concatenate([old_token_weights, new_weights], axis=0)
```

새로운 TextEncoder를 구성하고 준비해보겠습니다.

```python
# download_weights를 False로 설정해야, 초기화할 수 있습니다. (그렇지 않으면, 가중치를 로드하려고 시도합니다)
new_encoder = keras_cv.models.stable_diffusion.TextEncoder(
    keras_cv.models.stable_diffusion.stable_diffusion.MAX_PROMPT_LENGTH,
    vocab_size=new_vocab_size,
    download_weights=False,
)
for index, layer in enumerate(stable_diffusion.text_encoder.layers):
    # 레이어 2는 임베딩 레이어이므로, 가중치 복사에서 제외합니다.
    if index == 2:
        continue
    new_encoder.layers[index].set_weights(layer.get_weights())


new_encoder.layers[2].token_embedding.set_weights([new_weights])
new_encoder.layers[2].position_embedding.set_weights(old_position_weights)

stable_diffusion._text_encoder = new_encoder
stable_diffusion._text_encoder.compile(jit_compile=True)
```

## 트레이닝 {#training}

이제 흥미로운 부분인 트레이닝으로 넘어갈 수 있습니다!

TextualInversion에서 트레이닝되는 모델의 유일한 부분은 임베딩 벡터입니다. 나머지 모델을 동결해 보겠습니다.

```python
stable_diffusion.diffusion_model.trainable = False
stable_diffusion.decoder.trainable = False
stable_diffusion.text_encoder.trainable = True

stable_diffusion.text_encoder.layers[2].trainable = True


def traverse_layers(layer):
    if hasattr(layer, "layers"):
        for layer in layer.layers:
            yield layer
    if hasattr(layer, "token_embedding"):
        yield layer.token_embedding
    if hasattr(layer, "position_embedding"):
        yield layer.position_embedding


for layer in traverse_layers(stable_diffusion.text_encoder):
    if isinstance(layer, keras.layers.Embedding) or "clip_embedding" in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False

new_encoder.layers[2].position_embedding.trainable = False
```

적절한 가중치가 trainable 설정되었는지 확인해 보겠습니다.

```python
all_models = [
    stable_diffusion.text_encoder,
    stable_diffusion.diffusion_model,
    stable_diffusion.decoder,
]
print([[w.shape for w in model.trainable_weights] for model in all_models])
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
[[TensorShape([49409, 768])], [], []]
```

{{% /details %}}

## 새로운 임베딩 트레이닝 {#training-the-new-embedding}

임베딩을 트레이닝하기 위해, 몇 가지 유틸리티가 필요합니다.
KerasCV에서 NoiseScheduler를 import하고, 아래 유틸리티를 정의합니다.

- `sample_from_encoder_outputs`는 평균만 취하는 것이 아니라(다른 많은 SD 애플리케이션처럼),
  이미지 인코더에서 생성된 통계적 분포에서 샘플링하는 베이스 StableDiffusion 이미지 인코더를 둘러싼 래퍼입니다.
- `get_timestep_embedding`은 확산 모델에 대한 지정된 타임스텝에 대한 임베딩을 생성합니다.
- `get_position_ids`는 텍스트 인코더에 대한 위치 ID의 텐서를 생성합니다.
  (이것은 `[1, MAX_PROMPT_LENGTH]`의 시리즈일 뿐입니다.)

```python
# 인코더에서 최상위 레이어를 제거하면, 분산이 차단(cuts off)되고 평균만 반환됩니다.
training_image_encoder = keras.Model(
    stable_diffusion.image_encoder.input,
    stable_diffusion.image_encoder.layers[-2].output,
)


def sample_from_encoder_outputs(outputs):
    mean, logvar = tf.split(outputs, 2, axis=-1)
    logvar = tf.clip_by_value(logvar, -30.0, 20.0)
    std = tf.exp(0.5 * logvar)
    sample = tf.random.normal(tf.shape(mean))
    return mean + std * sample


def get_timestep_embedding(timestep, dim=320, max_period=10000):
    half = dim // 2
    freqs = tf.math.exp(
        -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
    )
    args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
    embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
    return embedding


def get_position_ids():
    return tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)
```

다음으로, `StableDiffusionFineTuner`를 구현합니다.
이는 `train_step`을 재정의하여, 텍스트 인코더의 토큰 임베딩을 트레이닝시키는,
[`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}})의 하위 클래스입니다.
이는 Textual Inversion 알고리즘의 핵심입니다.

추상적으로 말해서, 트레이닝 단계는 트레이닝 이미지에 대한 동결된 SD 이미지 인코더의 잠재 분포 출력에서 ​​샘플을 가져오고,
해당 샘플에 노이즈를 추가한 다음, 해당 노이즈 샘플을 동결된 확산 모델에 전달합니다.
확산 모델의 숨겨진 상태는 이미지에 해당하는 프롬프트에 대한 텍스트 인코더의 출력입니다.

최종 목표 상태는 확산 모델이 텍스트 인코딩을 숨겨진 상태로 사용하여, 샘플에서 노이즈를 분리할 수 있다는 것입니다.
따라서, 손실은 노이즈의 평균 제곱 오차와 확산 모델의 출력(이상적으로는 노이즈에서 이미지 잠재를 제거함)입니다.

우리는 텍스트 인코더의 토큰 임베딩에 대해서만 그래디언트를 계산하고,
트레이닝 단계에서는 우리가 학습하는 토큰을 제외한 모든 토큰에 대한 그래디언트를 0으로 만듭니다.

트레이닝 단계에 대한 자세한 내용은 인라인 코드 주석을 참조하세요.

```python
class StableDiffusionFineTuner(keras.Model):
    def __init__(self, stable_diffusion, noise_scheduler, **kwargs):
        super().__init__(**kwargs)
        self.stable_diffusion = stable_diffusion
        self.noise_scheduler = noise_scheduler

    def train_step(self, data):
        images, embeddings = data

        with tf.GradientTape() as tape:
            # 트레이닝 이미지에 대한 예측 분포의 샘플
            latents = sample_from_encoder_outputs(training_image_encoder(images))
            # 잠재값은 StableDiffusion의 트레이닝에 사용된 잠재값의 규모와 일치하도록 다운샘플링되어야 합니다.
            # 이 숫자는 실제로 모델을 트레이닝할 때 선택한 "마법" 상수일 뿐입니다.
            latents = latents * 0.18215

            # 잠재 샘플과 동일한 모양으로 랜덤 노이즈를 생성합니다.
            noise = tf.random.normal(tf.shape(latents))
            batch_dim = tf.shape(latents)[0]

            # 배치의 각 샘플에 대해 랜덤 시간 단계를 선택합니다.
            timesteps = tf.random.uniform(
                (batch_dim,),
                minval=0,
                maxval=noise_scheduler.train_timesteps,
                dtype=tf.int64,
            )

            # 각 샘플의 타임스텝에 따라 잠재 데이터(latents)에 노이즈를 추가합니다.
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # 확산 모델에서 숨겨진 상태로 사용하기 위해, 트레이닝 샘플의 텍스트를 인코딩합니다.
            encoder_hidden_state = self.stable_diffusion.text_encoder(
                [embeddings, get_position_ids()]
            )

            # 배치의 각 샘플에 대해 무작위로 선택된 타임스텝에 대한 타임스텝 임베딩을 계산합니다.
            timestep_embeddings = tf.map_fn(
                fn=get_timestep_embedding,
                elems=timesteps,
                fn_output_signature=tf.float32,
            )

            # 확산 모델을 호출합니다
            noise_pred = self.stable_diffusion.diffusion_model(
                [noisy_latents, timestep_embeddings, encoder_hidden_state]
            )

            # 평균 제곱 오차 손실을 계산하고, 이를 줄입니다.
            loss = self.compiled_loss(noise_pred, noise)
            loss = tf.reduce_mean(loss, axis=2)
            loss = tf.reduce_mean(loss, axis=1)
            loss = tf.reduce_mean(loss)

        # trainable 가중치를 로드하고, 해당 가중치에 대한 그래디언트를 계산합니다.
        trainable_weights = self.stable_diffusion.text_encoder.trainable_weights
        grads = tape.gradient(loss, trainable_weights)

        # 그래디언트는 인덱스가 지정된 슬라이스에 저장되므로,
        # 플레이스홀더 토큰이 포함된 슬라이스의 인덱스를 찾아야 합니다.
        index_of_placeholder_token = tf.reshape(tf.where(grads[0].indices == 49408), ())
        condition = grads[0].indices == 49408
        condition = tf.expand_dims(condition, axis=-1)

        # 그래디언트를 재정의하고, 플레이스홀더 토큰이 아닌 모든 슬라이스의 그래디언트를 0으로 만들어서,
        # 다른 모든 토큰의 가중치를 효과적으로 동결합니다.
        grads[0] = tf.IndexedSlices(
            values=tf.where(condition, grads[0].values, 0),
            indices=grads[0].indices,
            dense_shape=grads[0].dense_shape,
        )

        self.optimizer.apply_gradients(zip(grads, trainable_weights))
        return {"loss": loss}
```

트레이닝을 시작하기 전에, StableDiffusion이 토큰에 대해 어떤 결과를 생성하는지 살펴보겠습니다.

```python
generated = stable_diffusion.text_to_image(
    f"an oil painting of {placeholder_token}", seed=1337, batch_size=3
)
plot_images(generated)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
25/25 [==============================] - 19s 314ms/step
```

{{% /details %}}

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_33_1.png)

보시다시피, 모델은 여전히 ​​토큰을 고양이로 생각합니다.
이는 커스텀 토큰을 초기화하는 데 사용한, 시드 토큰이기 때문입니다.

이제, 트레이닝을 시작하려면, 다른 Keras 모델과 마찬가지로 모델을 `compile()`하면 됩니다.
그렇게 하기 전에, 트레이닝을 위한 노이즈 스케줄러를 인스턴스화하고,
학습률 및 옵티마이저와 같은 트레이닝 매개변수를 구성합니다.

```python
noise_scheduler = NoiseScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    train_timesteps=1000,
)
trainer = StableDiffusionFineTuner(stable_diffusion, noise_scheduler, name="trainer")
EPOCHS = 50
learning_rate = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4, decay_steps=train_ds.cardinality() * EPOCHS
)
optimizer = keras.optimizers.Adam(
    weight_decay=0.004, learning_rate=learning_rate, epsilon=1e-8, global_clipnorm=10
)

trainer.compile(
    optimizer=optimizer,
    # 트레이닝 단계에서 수동으로 축소(reduction)를 수행하므로, 여기서는 none이 요구됩니다.
    loss=keras.losses.MeanSquaredError(reduction="none"),
)
```

트레이닝을 모니터링하기 위해,
우리는 커스텀 토큰을 사용하여 매 에포크마다 몇 개의 이미지를 생성하는 [`keras.callbacks.Callback`]({{< relref "/docs/api/callbacks/base_callback#callback-class" >}})를 생성할 수 있습니다.

우리는 서로 다른 프롬프트를 사용하여 세 개의 콜백을 생성하여, 트레이닝 과정에서 어떻게 진행되는지 볼 수 있습니다.
우리는 고정된 시드를 사용하여, 트레이닝된 토큰의 진행 상황을 쉽게 볼 수 있습니다.

```python
class GenerateImages(keras.callbacks.Callback):
    def __init__(
        self, stable_diffusion, prompt, steps=50, frequency=10, seed=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.stable_diffusion = stable_diffusion
        self.prompt = prompt
        self.seed = seed
        self.frequency = frequency
        self.steps = steps

    def on_epoch_end(self, epoch, logs):
        if epoch % self.frequency == 0:
            images = self.stable_diffusion.text_to_image(
                self.prompt, batch_size=3, num_steps=self.steps, seed=self.seed
            )
            plot_images(
                images,
            )


cbs = [
    GenerateImages(
        stable_diffusion, prompt=f"an oil painting of {placeholder_token}", seed=1337
    ),
    GenerateImages(
        stable_diffusion, prompt=f"gandalf the gray as a {placeholder_token}", seed=1337
    ),
    GenerateImages(
        stable_diffusion,
        prompt=f"two {placeholder_token} getting married, photorealistic, high quality",
        seed=1337,
    ),
]
```

이제, `model.fit()`를 호출하는 것만 남았습니다!

```python
trainer.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=cbs,
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Epoch 1/50
50/50 [==============================] - 16s 318ms/step
50/50 [==============================] - 16s 318ms/step
50/50 [==============================] - 16s 318ms/step
250/250 [==============================] - 194s 469ms/step - loss: 0.1533
Epoch 2/50
250/250 [==============================] - 68s 269ms/step - loss: 0.1557
Epoch 3/50
250/250 [==============================] - 68s 269ms/step - loss: 0.1359
Epoch 4/50
250/250 [==============================] - 68s 269ms/step - loss: 0.1693
Epoch 5/50
250/250 [==============================] - 68s 269ms/step - loss: 0.1475
Epoch 6/50
250/250 [==============================] - 68s 268ms/step - loss: 0.1472
Epoch 7/50
250/250 [==============================] - 68s 268ms/step - loss: 0.1533
Epoch 8/50
250/250 [==============================] - 68s 268ms/step - loss: 0.1450
Epoch 9/50
250/250 [==============================] - 68s 268ms/step - loss: 0.1639
Epoch 10/50
250/250 [==============================] - 68s 269ms/step - loss: 0.1351
Epoch 11/50
50/50 [==============================] - 16s 316ms/step
50/50 [==============================] - 16s 316ms/step
50/50 [==============================] - 16s 317ms/step
250/250 [==============================] - 116s 464ms/step - loss: 0.1474
Epoch 12/50
250/250 [==============================] - 68s 268ms/step - loss: 0.1737
Epoch 13/50
250/250 [==============================] - 68s 269ms/step - loss: 0.1427
Epoch 14/50
250/250 [==============================] - 68s 269ms/step - loss: 0.1698
Epoch 15/50
250/250 [==============================] - 68s 270ms/step - loss: 0.1424
Epoch 16/50
250/250 [==============================] - 68s 268ms/step - loss: 0.1339
Epoch 17/50
250/250 [==============================] - 68s 268ms/step - loss: 0.1397
Epoch 18/50
250/250 [==============================] - 68s 268ms/step - loss: 0.1469
Epoch 19/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1649
Epoch 20/50
250/250 [==============================] - 68s 268ms/step - loss: 0.1582
Epoch 21/50
50/50 [==============================] - 16s 315ms/step
50/50 [==============================] - 16s 316ms/step
50/50 [==============================] - 16s 316ms/step
250/250 [==============================] - 116s 462ms/step - loss: 0.1331
Epoch 22/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1319
Epoch 23/50
250/250 [==============================] - 68s 267ms/step - loss: 0.1521
Epoch 24/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1486
Epoch 25/50
250/250 [==============================] - 68s 267ms/step - loss: 0.1449
Epoch 26/50
250/250 [==============================] - 67s 266ms/step - loss: 0.1349
Epoch 27/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1454
Epoch 28/50
250/250 [==============================] - 68s 268ms/step - loss: 0.1394
Epoch 29/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1489
Epoch 30/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1338
Epoch 31/50
50/50 [==============================] - 16s 315ms/step
50/50 [==============================] - 16s 320ms/step
50/50 [==============================] - 16s 315ms/step
250/250 [==============================] - 116s 462ms/step - loss: 0.1328
Epoch 32/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1693
Epoch 33/50
250/250 [==============================] - 67s 266ms/step - loss: 0.1420
Epoch 34/50
250/250 [==============================] - 67s 266ms/step - loss: 0.1255
Epoch 35/50
250/250 [==============================] - 67s 266ms/step - loss: 0.1239
Epoch 36/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1558
Epoch 37/50
250/250 [==============================] - 68s 267ms/step - loss: 0.1527
Epoch 38/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1461
Epoch 39/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1555
Epoch 40/50
250/250 [==============================] - 67s 266ms/step - loss: 0.1515
Epoch 41/50
50/50 [==============================] - 16s 315ms/step
50/50 [==============================] - 16s 315ms/step
50/50 [==============================] - 16s 315ms/step
250/250 [==============================] - 116s 461ms/step - loss: 0.1291
Epoch 42/50
250/250 [==============================] - 67s 266ms/step - loss: 0.1474
Epoch 43/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1908
Epoch 44/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1506
Epoch 45/50
250/250 [==============================] - 68s 267ms/step - loss: 0.1424
Epoch 46/50
250/250 [==============================] - 67s 266ms/step - loss: 0.1601
Epoch 47/50
250/250 [==============================] - 67s 266ms/step - loss: 0.1312
Epoch 48/50
250/250 [==============================] - 67s 266ms/step - loss: 0.1524
Epoch 49/50
250/250 [==============================] - 67s 266ms/step - loss: 0.1477
Epoch 50/50
250/250 [==============================] - 67s 267ms/step - loss: 0.1397

<keras.callbacks.History at 0x7f183aea3eb8>
```

{{% /details %}}

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_2.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_3.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_4.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_5.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_6.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_7.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_8.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_9.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_10.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_11.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_12.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_13.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_14.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_15.png)

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_39_16.png)

모델이 시간이 지남에 따라 새로운 토큰을 어떻게 학습하는지 보는 것은 꽤 재밌습니다.
이것으로 놀면서, 어떻게 트레이닝 매개변수와 트레이닝 데이터 세트를 조정하여,
최상의 이미지를 생성할 수 있는지 살펴보세요!

## 미세 조정된 모델을 돌려보세요 {#taking-the-fine-tuned-model-for-a-spin}

이제 정말 재밌는 부분입니다.
커스텀 토큰에 대한 토큰 임베딩을 배웠으므로,
이제 다른 토큰과 같은 방식으로 StableDiffusion으로 이미지를 생성할 수 있습니다!

고양이 인형 토큰의 샘플 출력과 함께 시작하는 데 도움이 되는, 몇 가지 재미있는 예시 프롬프트가 있습니다!

```python
generated = stable_diffusion.text_to_image(
    f"Gandalf as a {placeholder_token} fantasy art drawn by disney concept artists, "
    "golden colour, high quality, highly detailed, elegant, sharp focus, concept art, "
    "character concepts, digital painting, mystery, adventure",
    batch_size=3,
)
plot_images(generated)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
25/25 [==============================] - 8s 316ms/step
```

{{% /details %}}

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_42_1.png)

```python
generated = stable_diffusion.text_to_image(
    f"A masterpiece of a {placeholder_token} crying out to the heavens. "
    f"Behind the {placeholder_token}, an dark, evil shade looms over it - sucking the "
    "life right out of it.",
    batch_size=3,
)
plot_images(generated)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
25/25 [==============================] - 8s 314ms/step
```

{{% /details %}}

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_43_1.png)

```python
generated = stable_diffusion.text_to_image(
    f"An evil {placeholder_token}.", batch_size=3
)
plot_images(generated)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
25/25 [==============================] - 8s 322ms/step
```

{{% /details %}}

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_44_1.png)

```python
generated = stable_diffusion.text_to_image(
    f"A mysterious {placeholder_token} approaches the great pyramids of egypt.",
    batch_size=3,
)
plot_images(generated)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
25/25 [==============================] - 8s 315ms/step
```

{{% /details %}}

![png](/images/examples/generative/fine_tune_via_textual_inversion/fine_tune_via_textual_inversion_45_1.png)

## 결론 {#conclusions}

Textual Inversion 알고리즘을 사용하면 StableDiffusion에 새로운 개념을 가르칠 수 있습니다!

다음으로 따라야 할 몇 가지 단계:

- 직접 프롬프트를 시도해 보세요.
- 모델에 스타일을 가르쳐 주세요.
- 좋아하는 애완 고양이 또는 개의 데이터 세트를 수집하여 모델에 대해 가르쳐 주세요.
