---
title: Stable Diffusion 미세 조정
linkTitle: Stable Diffusion 미세 조정
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-23" >}}

**{{< t f_author >}}** [Sayak Paul](https://twitter.com/RisingSayak), [Chansung Park](https://twitter.com/algo_diver)  
**{{< t f_date_created >}}** 2022/12/28  
**{{< t f_last_modified >}}** 2023/01/13  
**{{< t f_description >}}** 커스텀 이미지 캡션 데이터 세트를 사용하여 Stable Diffusion을 미세 조정합니다.

{{< keras/version v=2 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/finetune_stable_diffusion.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/finetune_stable_diffusion.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

이 튜토리얼은 `{image, caption}` 쌍의 커스텀 데이터 세트에서 [Stable Diffusion 모델]({{< relref "/docs/guides/keras_cv/generate_images_with_stable_diffusion" >}})을 미세 조정하는 방법을 보여줍니다.
우리는 Hugging Face [여기](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)에서 제공하는 미세 조정 스크립트를 기반으로 구축합니다.

우리는 당신이 Stable Diffusion 모델에 대한 높은 수준의 이해를 가지고 있다고 가정합니다.
다음 리소스는 이와 관련하여 더 많은 정보를 찾는 데 도움이 될 수 있습니다.

- [KerasCV에서 Stable Diffusion을 사용한 고성능 이미지 생성]({{< relref "/docs/guides/keras_cv/generate_images_with_stable_diffusion" >}})
- [디퓨저를 사용한 Stable Diffusion](https://huggingface.co/blog/stable_diffusion)

코드를 실행하려면 최소 30GB의 메모리가 있는 GPU를 사용하는 것이 좋습니다.

가이드를 마치면, 흥미로운 포켓몬 이미지를 생성할 수 있을 것입니다.

![custom-pokemons](/images/examples/generative/finetune_stable_diffusion/X4m614M.png)

이 튜토리얼은 KerasCV 0.4.0에 의존합니다.
또한 혼합 정밀도로 AdamW를 사용하려면, 최소 TensorFlow 2.11이 필요합니다.

```python
!pip install keras-cv==0.6.0 -q
!pip install -U tensorflow -q
!pip install keras-core -q
```

## 우리는 무엇을 미세 조정하고 있나요? {#what-are-we-fine-tuning}

Stable Diffusion 모델은 몇 가지 핵심 모델로 분해될 수 있습니다.

- 입력 프롬프트를 잠재 공간에 프로젝션하는 텍스트 인코더. (이미지와 관련된 캡션을 "프롬프트"라고 합니다.)
- 입력 이미지를 이미지 벡터 공간으로 작용하는 잠재 공간에 투사하는 변형 자동 인코더(VAE, variational autoencoder).
- 잠재 벡터를 정제하고 인코딩된 텍스트 프롬프트에 따라 다른 잠재 벡터를 생성하는 디퓨전 모델
- 디퓨전 모델에서 잠재 벡터가 주어지면 이미지를 생성하는 디코더.

텍스트 프롬프트에서 이미지를 생성하는 과정에서는,
일반적으로 이미지 인코더가 사용되지 않는다는 점에 유의해야 합니다.

그러나, 미세 조정 과정에서 워크플로는 다음과 같습니다.

1. 입력 텍스트 프롬프트는 텍스트 인코더에 의해 잠재 공간에 프로젝션됩니다.
2. 입력 이미지는 VAE의 이미지 인코더 부분에 의해 잠재 공간에 프로젝션됩니다.
3. 주어진 시간 단계에 대한 이미지 잠재 벡터에 소량의 노이즈가 추가됩니다.
4. 디퓨전 모델은 이 두 공간의 잠재 벡터와 시간 단계 임베딩을 사용하여, 이미지 잠재에 추가된 노이즈를 예측합니다.
5. 예측된 노이즈와 3단계에서 추가된 원래 노이즈 사이에서 재구성 손실을 계산합니다.
6. 마지막으로, 디퓨전 모델 매개변수는 경사 하강법을 사용하여 이 손실과 관련하여 최적화됩니다.

미세 조정 중에 디퓨전 모델 매개변수만 업데이트되고,
(사전 트레이닝된) 텍스트와 이미지 인코더는 고정된 상태로 유지됩니다.

이것이 복잡하게 들리더라도 걱정하지 마십시오. 코드는 이것보다 훨씬 간단합니다!

## Imports {#imports}

```python
from textwrap import wrap
import os

import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from tensorflow import keras
```

## 데이터 로딩 {#data-loading}

우리는 [포켓몬 BLIP 캡션](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions) 데이터 세트를 사용합니다.
하지만, 우리는 [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data)에 더 잘 맞도록,
원본 데이터 세트에서 파생된 약간 다른 버전을 사용할 것입니다.
자세한 내용은 [문서](https://huggingface.co/datasets/sayakpaul/pokemon-blip-original-version)를 참조하세요.

```python
data_path = tf.keras.utils.get_file(
    origin="https://huggingface.co/datasets/sayakpaul/pokemon-blip-original-version/resolve/main/pokemon_dataset.tar.gz",
    untar=True,
)

data_frame = pd.read_csv(os.path.join(data_path, "data.csv"))

data_frame["image_path"] = data_frame["image_path"].apply(
    lambda x: os.path.join(data_path, x)
)
data_frame.head()
```

|     | image_path                                        | caption                                           |
| --- | ------------------------------------------------- | ------------------------------------------------- |
| 0   | /home/jupyter/.keras/datasets/pokemon_dataset/... | a drawing of a green pokemon with red eyes        |
| 1   | /home/jupyter/.keras/datasets/pokemon_dataset/... | a green and yellow toy with a red nose            |
| 2   | /home/jupyter/.keras/datasets/pokemon_dataset/... | a red and white ball with an angry look on its... |
| 3   | /home/jupyter/.keras/datasets/pokemon_dataset/... | a cartoon ball with a smile on it's face          |
| 4   | /home/jupyter/.keras/datasets/pokemon_dataset/... | a bunch of balls with faces drawn on them         |

`{image, caption}` 쌍이 833개뿐이므로,
캡션에서 텍스트 임베딩을 미리 계산할 수 있습니다.
게다가, 텍스트 인코더는 미세 조정 과정에서 동결되므로,
이렇게 하면 계산을 약간 절약할 수 있습니다.

텍스트 인코더를 사용하기 전에, 캡션을 토큰화해야 합니다.

```python
# 패딩 토큰과 최대 프롬프트 길이는 텍스트 인코더에 따라 다릅니다.
# 다른 텍스트 인코더를 사용하는 경우, 이에 따라 변경해야 합니다.
PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77

# 토크나이저를 로드합니다.
tokenizer = SimpleTokenizer()

# 토큰을 토큰화하고 패딩하는 메서드.
def process_text(caption):
    tokens = tokenizer.encode(caption)
    tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
    return np.array(tokens)


# 토큰화된 캡션을 배열로 정리합니다.
tokenized_texts = np.empty((len(data_frame), MAX_PROMPT_LENGTH))

all_captions = list(data_frame["caption"].values)
for i, caption in enumerate(all_captions):
    tokenized_texts[i] = process_text(caption)
```

## [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) 준비 {#tfdatadataset}

이 섹션에서는, 입력 이미지 파일 경로와 해당 캡션 토큰에서
[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) 객체를 준비합니다.
이 섹션에는 다음이 포함됩니다.

- 토큰화된 캡션에서 텍스트 임베딩을 사전 계산합니다.
- 입력 이미지의 로딩 및 보강.
- 데이터 세트의 셔플 및 배치(batching).

```python
RESOLUTION = 256
AUTO = tf.data.AUTOTUNE
POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)

augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.CenterCrop(RESOLUTION, RESOLUTION),
        keras_cv.layers.RandomFlip(),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)
text_encoder = TextEncoder(MAX_PROMPT_LENGTH)


def process_image(image_path, tokenized_text):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, 3)
    image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
    return image, tokenized_text


def apply_augmentation(image_batch, token_batch):
    return augmenter(image_batch), token_batch


def run_text_encoder(image_batch, token_batch):
    return (
        image_batch,
        token_batch,
        text_encoder([token_batch, POS_IDS], training=False),
    )


def prepare_dict(image_batch, token_batch, encoded_text_batch):
    return {
        "images": image_batch,
        "tokens": token_batch,
        "encoded_text": encoded_text_batch,
    }


def prepare_dataset(image_paths, tokenized_texts, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, tokenized_texts))
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(process_image, num_parallel_calls=AUTO).batch(batch_size)
    dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTO)
    dataset = dataset.map(run_text_encoder, num_parallel_calls=AUTO)
    dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO)
    return dataset.prefetch(AUTO)
```

베이스라인 Stable Diffusion 모델은 512x512 해상도의 이미지를 사용하여 트레이닝되었습니다.
고해상도 이미지를 사용하여 트레이닝된 모델이 저해상도 이미지로 잘 전환될 가능성은 낮습니다.
그러나, 현재 모델은 해상도를 512x512로 유지하면(혼합 정밀도를 활성화하지 않고) OOM으로 이어질 것입니다.
따라서, 대화형 데모의 이익을 위해, 입력 해상도를 256x256으로 유지했습니다.

```python
# 데이터 세트를 준비합니다.
training_dataset = prepare_dataset(
    np.array(data_frame["image_path"]), tokenized_texts, batch_size=4
)

# 샘플 배치를 가져와 조사해 보세요.
sample_batch = next(iter(training_dataset))

for k in sample_batch:
    print(k, sample_batch[k].shape)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
images (4, 256, 256, 3)
tokens (4, 77)
encoded_text (4, 77, 768)
```

{{% /details %}}

또한 트레이닝 이미지와 해당 캡션을 살펴볼 수도 있습니다.

```python
plt.figure(figsize=(20, 10))

for i in range(3):
    ax = plt.subplot(1, 4, i + 1)
    plt.imshow((sample_batch["images"][i] + 1) / 2)

    text = tokenizer.decode(sample_batch["tokens"][i].numpy().squeeze())
    text = text.replace("<|startoftext|>", "")
    text = text.replace("<|endoftext|>", "")
    text = "\n".join(wrap(text, 12))
    plt.title(text, fontsize=15)

    plt.axis("off")
```

![png](/images/examples/generative/finetune_stable_diffusion/finetune_stable_diffusion_15_0.png)

## 파인튜닝 루프를 위한 트레이너 클래스 {#a-trainer-class-for-the-fine-tuning-loop}

```python
class Trainer(tf.keras.Model):
    # 참조:
    # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

    def __init__(
        self,
        diffusion_model,
        vae,
        noise_scheduler,
        use_mixed_precision=False,
        max_grad_norm=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.diffusion_model = diffusion_model
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.max_grad_norm = max_grad_norm

        self.use_mixed_precision = use_mixed_precision
        self.vae.trainable = False

    def train_step(self, inputs):
        images = inputs["images"]
        encoded_text = inputs["encoded_text"]
        batch_size = tf.shape(images)[0]

        with tf.GradientTape() as tape:
            # 잠재 공간에 이미지를 프로젝션하고 샘플을 추출합니다.
            latents = self.sample_from_encoder_outputs(self.vae(images, training=False))
            # 여기서 마법의 숫자에 대해 자세히 알아보세요:
            # https://keras.io/examples/generative/fine_tune_via_textual_inversion/
            latents = latents * 0.18215

            # 잠재 데이터에 추가할 샘플 노이즈입니다.
            noise = tf.random.normal(tf.shape(latents))

            # 각 이미지에 대해 랜덤 타임스텝을 샘플링합니다.
            timesteps = tnp.random.randint(
                0, self.noise_scheduler.train_timesteps, (batch_size,)
            )

            # 각 타임스텝의 노이즈 크기에 따라 잠재 노이즈를 추가합니다. (이것은 전방 확산 과정입니다)
            noisy_latents = self.noise_scheduler.add_noise(
                tf.cast(latents, noise.dtype), noise, timesteps
            )

            # 지금은 샘플링된 노이즈에 따라 예측 타입에 따른 손실 대상을 구합니다.
            target = noise  # noise_schedule.predict_epsilon == True

            # residual 노이즈를 예측하고, 손실을 계산합니다.
            timestep_embedding = tf.map_fn(
                lambda t: self.get_timestep_embedding(t), timesteps, dtype=tf.float32
            )
            timestep_embedding = tf.squeeze(timestep_embedding, 1)
            model_pred = self.diffusion_model(
                [noisy_latents, timestep_embedding, encoded_text], training=True
            )
            loss = self.compiled_loss(target, model_pred)
            if self.use_mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        # 확산 모델의 매개변수를 업데이트합니다.
        trainable_vars = self.diffusion_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        if self.use_mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {m.name: m.result() for m in self.metrics}

    def get_timestep_embedding(self, timestep, dim=320, max_period=10000):
        half = dim // 2
        log_max_period = tf.math.log(tf.cast(max_period, tf.float32))
        freqs = tf.math.exp(
            -log_max_period * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return embedding

    def sample_from_encoder_outputs(self, outputs):
        mean, logvar = tf.split(outputs, 2, axis=-1)
        logvar = tf.clip_by_value(logvar, -30.0, 20.0)
        std = tf.exp(0.5 * logvar)
        sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + std * sample

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        # 이 메서드를 재정의하면, 이 트레이너 클래스에서
        # `ModelCheckpoint` 콜백을 직접 사용할 수 있습니다.
        # 이 경우, 미세 조정 중에 트레이닝하는 것이 `diffusion_model`이므로,
        # `diffusion_model`만 체크포인트합니다.
        self.diffusion_model.save_weights(
            filepath=filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
```

여기서 주목해야 할 중요한 구현 세부 사항 하나:
이미지 인코더(VAE)에서 생성된 잠재 벡터를 직접 취하는 대신,
이미지 인코더에서 예측한 평균과 로그 분산에서 샘플링합니다.
이런 방식으로, 더 나은 샘플 품질과 다양성을 얻을 수 있습니다.

이러한 모델을 미세 조정하기 위해,
모델 가중치의 지수 이동 평균과 함께 혼합 정밀도 학습에 대한 지원을 추가하는 것이 일반적입니다.
그러나 간결함을 위해 이러한 요소를 버립니다.
이에 대한 자세한 내용은 튜토리얼 후반부에서 설명합니다.

## 트레이너를 초기화하고 컴파일 {#initialize-the-trainer-and-compile-it}

```python
# 기본 GPU에 텐서 코어가 있는 경우, 혼합 정밀도 트레이닝을 활성화합니다.
USE_MP = True
if USE_MP:
    keras.mixed_precision.set_global_policy("mixed_float16")

image_encoder = ImageEncoder()
diffusion_ft_trainer = Trainer(
    diffusion_model=DiffusionModel(RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH),
    # 인코더에서 최상위 레이어를 제거하면, 분산이 제거(cuts off)되고 평균만 반환됩니다.
    vae=tf.keras.Model(
        image_encoder.input,
        image_encoder.layers[-2].output,
    ),
    noise_scheduler=NoiseScheduler(),
    use_mixed_precision=USE_MP,
)

# 이러한 하이퍼파라미터는 Hugging Face의 이 튜토리얼에서 나왔습니다.
# https://huggingface.co/docs/diffusers/training/text2image
lr = 1e-5
beta_1, beta_2 = 0.9, 0.999
weight_decay = (1e-2,)
epsilon = 1e-08

optimizer = tf.keras.optimizers.experimental.AdamW(
    learning_rate=lr,
    weight_decay=weight_decay,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
)
diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")
```

## 미세 조정 {#fine-tuning}

이 튜토리얼의 실행 시간을 짧게 유지하기 위해, 에포크에 맞춰 미세 조정만 했습니다.

```python
epochs = 1
ckpt_path = "finetuned_stable_diffusion.h5"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    save_weights_only=True,
    monitor="loss",
    mode="min",
)
diffusion_ft_trainer.fit(training_dataset, epochs=epochs, callbacks=[ckpt_callback])
```

## 추론 {#inference}

우리는 512x512의 이미지 해상도에서 60에포크 동안 모델을 미세 조정했습니다.
이 해상도로 트레이닝할 수 있도록 혼합 정밀도 지원을 통합했습니다.
자세한 내용은 [이 저장소](https://github.com/sayakpaul/stabe-diffusion-keras-ft)를 확인하세요.
또한 미세 조정된 모델 매개변수의 지수 이동 평균화와 모델 체크포인팅에 대한 지원도 제공합니다.

이 섹션에서는, 미세 조정 60 에포크 후 파생된 체크포인트를 사용합니다.

```python
weights_path = tf.keras.utils.get_file(
    origin="https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned/resolve/main/ckpt_epochs_72_res_512_mp_True.h5"
)

img_height = img_width = 512
pokemon_model = keras_cv.models.StableDiffusion(
    img_width=img_width, img_height=img_height
)
# 우리는 미세 조정된 확산 모델의 가중치를 다시 로드합니다.
pokemon_model.diffusion_model.load_weights(weights_path)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
By using this model checkpoint, you acknowledge that its usage is subject to the terms of the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE
```

{{% /details %}}

이제, 이 모델을 시운전해 볼 수 있습니다.

```python
prompts = ["Yoda", "Hello Kitty", "A pokemon with red eyes"]
images_to_generate = 3
outputs = {}

for prompt in prompts:
    generated_images = pokemon_model.text_to_image(
        prompt, batch_size=images_to_generate, unconditional_guidance_scale=40
    )
    outputs.update({prompt: generated_images})
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
25/25 [==============================] - 17s 231ms/step
25/25 [==============================] - 6s 229ms/step
25/25 [==============================] - 6s 229ms/step
```

{{% /details %}}

60에포크의 미세 조정(적절한 수는 약 70)으로 생성된 이미지는 기준에 미치지 못했습니다.
그래서, 우리는 추론 시간 동안 Stable Diffusion이 취하는 단계 수와
`unconditional_guidance_scale` 매개변수를 실험했습니다.

우리는 `unconditional_guidance_scale`을 40으로 설정한 이 체크포인트에서 가장 좋은 결과를 발견했습니다.

```python
def plot_images(images, title):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(title, fontsize=12)
        plt.axis("off")


for prompt in outputs:
    plot_images(outputs[prompt], prompt)
```

![png](/images/examples/generative/finetune_stable_diffusion/finetune_stable_diffusion_28_0.png)

![png](/images/examples/generative/finetune_stable_diffusion/finetune_stable_diffusion_28_1.png)

![png](/images/examples/generative/finetune_stable_diffusion/finetune_stable_diffusion_28_2.png)

모델이 데이터 세트의 스타일에 적응하기 시작한 것을 알 수 있습니다.
더 많은 비교와 해설을 보려면,
[수반되는 저장소](https://github.com/sayakpaul/stable-diffusion-keras-ft#results)를 확인할 수 있습니다.
데모를 시도해 볼 모험심이 있다면,
[이 리소스](https://huggingface.co/spaces/sayakpaul/pokemon-sd-kerascv)를 확인할 수 있습니다.

## 결론 및 acknowledgements {#conclusion-and-acknowledgements}

커스텀 데이터 세트에서 Stable Diffusion 모델을 미세 조정하는 방법을 보여주었습니다.
결과가 미적으로 만족스럽지 않지만, 미세 조정의 에포크가 더 많아지면, 개선될 가능성이 있다고 생각합니다.
이를 가능하게 하려면, 그래디언트 축적 및 분산 트레이닝을 지원하는 것이 중요합니다.
이는 이 튜토리얼의 다음 단계로 생각할 수 있습니다.

Stable Diffusion 모델을 미세 조정할 수 있는 또 다른 흥미로운 방법이 있는데, textual inversion이라고 합니다.
자세한 내용은 [이 튜토리얼]({{< relref "/docs/examples/generative/fine_tune_via_textual_inversion" >}})을 참조하세요.

Google의 ML 개발자 프로그램 팀의 GCP 크레딧 지원에 감사드립니다.
[미세 조정 스크립트](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)를 제공해 주신 Hugging Face 팀에 감사드리고 싶습니다.
매우 읽기 쉽고 이해하기 쉽습니다.
