---
title: 노이즈 제거 디퓨전 확률론적 모델
linkTitle: 노이즈 제거 디퓨전 확률론적 모델
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-22" >}}

**{{< t f_author >}}** [A_K_Nain](https://twitter.com/A_K_Nain)  
**{{< t f_date_created >}}** 2022/11/30  
**{{< t f_last_modified >}}** 2022/12/07  
**{{< t f_description >}}** 노이즈 제거 디퓨전 확률론적 모델을 사용하여, 꽃 이미지를 생성합니다.

{{< keras/version v=2 >}}

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/ddpm.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/examples/generative/ddpm.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

생성 모델링은 지난 5년 동안 엄청난 성장을 경험했습니다.
VAE, GAN, 흐름 기반 모델과 같은 모델은, 특히 이미지에서,
고품질 콘텐츠를 생성하는 데 큰 성공을 거두었습니다.
디퓨전 모델은 이전 접근 방식보다 더 나은 것으로 입증된 새로운 타입의 생성 모델입니다.

디퓨전 모델은 비평형 열역학(non-equilibrium thermodynamics)에서 영감을 받았으며,
노이즈 제거를 통해 생성하는 방법을 학습합니다.
노이즈 제거를 통한 학습은 두 가지 프로세스로 구성되며, 각각은 마르코프 체인(Markov Chain)입니다.
이는 다음과 같습니다.

1. 순방향 프로세스:
   - 순방향 프로세스에서,
     일련의 시간 단계 `(t1, t2, ..., tn)`에서 데이터에 랜덤 노이즈를 천천히 추가합니다.
   - 현재 시간 단계의 샘플은 가우시안 분포에서 추출되며,
     분포의 평균은 이전 시간 단계의 샘플에 따라 조건이 지정되고,
     분포의 분산은 고정된 스케쥴을 따릅니다.
   - 순방향 프로세스가 끝나면, 샘플은 순수한 노이즈 분포로 끝납니다.
2. 역방향 프로세스:
   - 역방향 프로세스 동안, 우리는 모든 시간 단계에서 추가된 노이즈를 취소하려고 시도합니다.
   - 우리는 순수 노이즈 분포(순방향 프로세스의 마지막 단계)에서 시작하여,
   - 역방향 `(tn, tn-1, ..., t1)`에서 샘플의 노이즈를 제거하려고 시도합니다.

이 코드 예제에서는 [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) 논문 또는 약칭 DDPMs를 구현합니다.
이는 고품질 이미지를 생성하기 위해 디퓨전 모델을 사용하는 방법을 보여주는 최초의 논문이었습니다.
저자는 디퓨전 모델의 특정 매개변수화가,
트레이닝 중 여러 노이즈 레벨에 대한 denoising 점수 일치와,
샘플링 중 최상의 품질 결과를 생성하는 어닐링된 Langevin 역학(annealed Langevin dynamics)과,
동등함을 보여준다는 것을 증명했습니다.

이 논문은 디퓨전 프로세스에 관련된 두 가지 마르코프 체인(순방향 프로세스 및 역방향 프로세스)을 이미지에 대해 복제합니다.
순방향 프로세스는 고정되어 있으며, 논문에서 베타로 표시된 고정 분산 스케쥴에 따라,
이미지에 가우시안 노이즈를 점진적으로 추가합니다.
이는 이미지의 경우, 디퓨전 프로세스가 어떻게 보이는지 보여줍니다.
(image -> noise::noise -> image)

![diffusion process gif](/images/examples/generative/ddpm/Yn7tho9.gif)

이 논문에서는 두 가지 알고리즘을 설명합니다.
(1) 하나는 모델을 트레이닝하기 위한 알고리즘이고, (2) 다른 하나는 트레이닝된 모델에서 샘플링하기 위한 알고리즘입니다.
트레이닝은 음의 로그 가능도(negative log-likelihood)에 대한 일반적인 변분 경계(usual variational bound)를 최적화하여 수행됩니다.
목적 함수는 더욱 단순화되고, 네트워크는 노이즈 예측 네트워크로 처리됩니다.
최적화되면, 네트워크에서 샘플링하여 노이즈 샘플에서 새 이미지를 생성할 수 있습니다.
논문에 제시된 두 알고리즘에 대한 개요는 다음과 같습니다.

![ddpms](/images/examples/generative/ddpm/S7KH5hZ.png)

**참고:** DDPM은 디퓨전 모델을 구현하는 한 가지 방법일 뿐입니다.
또한 DDPM의 샘플링 알고리즘은 완전한 Markov chain을 복제합니다.
따라서, GAN과 같은 다른 생성 모델에 비해, 새로운 샘플을 생성하는 데 느립니다.
이 문제를 해결하기 위해, 많은 연구 노력이 이루어졌습니다.
그러한 예 중 하나는 Denoising Diffusion Implicit Models(약칭 DDIM)로,
저자는 마르코프 체인을 비마르코프(non-Markovian) 프로세스로 대체하여 더 빠르게 샘플링했습니다.
DDIM의 코드 예제는 [여기]({{< relref "/docs/examples/generative/ddim" >}})에서 찾을 수 있습니다.

DDPM 모델을 구현하는 것은 간단합니다.
(1) 이미지와 (2) 무작위로 샘플링된 시간 단계라는, 두 가지 입력을 사용하는 모델을 정의합니다.
각 트레이닝 단계에서, 다음 작업을 수행하여 모델을 트레이닝합니다.

1. 입력에 추가할 랜덤 노이즈를 샘플링합니다.
2. 포워드 프로세스를 적용하여, 샘플링된 노이즈로 입력을 디퓨전합니다.
3. 모델은 이러한 노이즈 샘플을 입력으로 사용하여, 각 시간 단계에 대한 노이즈 예측을 출력합니다.
4. 실제 노이즈와 예측된 노이즈가 주어지면, 손실 값을 계산합니다.
5. 그런 다음 그래디언트를 계산하고 모델 가중치를 업데이트합니다.

모델이 주어진 시간 단계에서 노이즈 샘플의 노이즈를 제거하는 방법을 알고 있으므로,
이 아이디어를 활용하여 순수 노이즈 분포에서 시작하여 새로운 샘플을 생성할 수 있습니다.

## 셋업 {#setup}

```python
import math
import numpy as np
import matplotlib.pyplot as plt

# GroupNormalization 레이어의 경우, TensorFlow 2.11 이상이 필요합니다.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
```

## 하이퍼파라미터 {#hyperparameters}

```python
batch_size = 32
num_epochs = 1  # 단지 시연을 위해서
total_timesteps = 1000
norm_groups = 8  # GroupNormalization 레이어에서 사용되는 그룹 수
learning_rate = 2e-4

img_size = 64
img_channels = 3
clip_min = -1.0
clip_max = 1.0

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2  # residual 블록 수

dataset_name = "oxford_flowers102"
splits = ["train"]
```

## 데이터 세트 {#dataset}

우리는 꽃 이미지를 생성하기 위해 [Oxford Flowers 102](https://www.tensorflow.org/datasets/catalog/oxford_flowers102) 데이터 세트를 사용합니다.
전처리 측면에서, 우리는 이미지를 원하는 이미지 크기로 조정하기 위해 중앙 자르기(center cropping)를 사용하고,
픽셀 값을 범위 `[-1.0, 1.0]`으로 재조정합니다.
이것은 [DDPMs 논문](https://arxiv.org/abs/2006.11239)의 저자가 적용한 픽셀 값의 범위와 일치합니다.
트레이닝 데이터를 보강하기 위해, 우리는 이미지를 무작위로 좌우로 뒤집습니다.

```python
# 데이터세트를 로드합니다
(ds,) = tfds.load(dataset_name, split=splits, with_info=False, shuffle_files=True)


def augment(img):
    """이미지를 무작위로 좌우로 뒤집습니다."""
    return tf.image.random_flip_left_right(img)


def resize_and_rescale(img, size):
    """먼저 이미지 크기를 원하는 크기로 조절한 다음,
    [-1.0, 1.0] 범위의 픽셀 값으로 재조정합니다.

    Args:
        img: Image 텐서
        size: 크기 조절을 위한 원하는 이미지 크기
    Returns:
        크기 조절(Resized) 및 재조정된(rescaled) 이미지 텐서
    """

    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    crop_size = tf.minimum(height, width)

    img = tf.image.crop_to_bounding_box(
        img,
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # 크기 조절
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, size=size, antialias=True)

    # 픽셀 값 재조정
    img = img / 127.5 - 1.0
    img = tf.clip_by_value(img, clip_min, clip_max)
    return img


def train_preprocessing(x):
    img = x["image"]
    img = resize_and_rescale(img, size=(img_size, img_size))
    img = augment(img)
    return img


train_ds = (
    ds.map(train_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size, drop_remainder=True)
    .shuffle(batch_size * 2)
    .prefetch(tf.data.AUTOTUNE)
)
```

## Gaussian 디퓨전 유틸리티 {#gaussian-diffusion-utilities}

우리는 포워드 프로세스와 리버스 프로세스를 별도의 유틸리티로 정의합니다.
이 유틸리티의 대부분 코드는 약간의 수정을 거쳐 원래 구현에서 빌려왔습니다.

```python
class GaussianDiffusion:
    """Gaussian 디퓨전 유틸리티

    Args:
        beta_start: 스케쥴된 분산의 시작 값
        beta_end: 스케쥴된 분산의 최종 값
        timesteps: 포워드 프로세스의 시간 단계 수
    """

    def __init__(
        self,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        clip_min=-1.0,
        clip_max=1.0,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # 선형 분산 스케쥴을 정의합니다
        self.betas = betas = np.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=np.float64,  # 더 나은 정밀도를 위해 float64 사용
        )
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # 디퓨전 q(x_t | x_{t-1}) 및 기타에 대한 계산
        self.sqrt_alphas_cumprod = tf.constant(
            np.sqrt(alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.log_one_minus_alphas_cumprod = tf.constant(
            np.log(1.0 - alphas_cumprod), dtype=tf.float32
        )

        self.sqrt_recip_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32
        )
        self.sqrt_recipm1_alphas_cumprod = tf.constant(
            np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32
        )

        # 사후(posterior) q(x_{t-1} | x_t, x_0)에 대한 계산
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

        # 디퓨전 체인의 시작 부분에서, 사후 분산(posterior variance)이 0이기 때문에,
        # 로그 계산이 잘렸습니다. (Log calculation clipped)
        self.posterior_log_variance_clipped = tf.constant(
            np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32
        )

        self.posterior_mean_coef1 = tf.constant(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            dtype=tf.float32,
        )

    def _extract(self, a, t, x_shape):
        """지정된 타임스텝에서 일부 계수를 추출한 다음,
        브로드캐스팅 목적으로 [batch_size, 1, 1, 1, 1, ...]로 모양을 변경(reshape)합니다.

        Args:
            a: 추출할 텐서
            t: 계수를 추출할 타임스텝
            x_shape: 현재 배치된 샘플의 모양
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_mean_variance(self, x_start, t):
        """현재 타임스텝의 평균과 분산을 추출합니다.

        Args:
            x_start: 초기 샘플(첫 번째 디퓨전 단계 이전)
            t: 현재 타임스텝
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start_shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """데이터를 디퓨전시킵니다.

        Args:
            x_start: 초기 샘플(첫 번째 디퓨전 단계 이전)
            t: 현재 타임스텝
            noise: 현재 타임스텝에 추가될 가우시안 노이즈
        Returns:
            시간 단계 `t`에서 디퓨전된 샘플
        """
        x_start_shape = tf.shape(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """디퓨전 사후(posterior) q(x_{t-1} | x_t, x_0)의 평균과 분산을 계산합니다.

        Args:
            x_start: 사후(posterior) 계산을 위한 시작점(샘플)
            x_t: `t` 타임스텝에서의 샘플
            t: 현재 타임스텝
        Returns:
            현재 타임스텝에서의 사후(Posterior) 평균 및 분산
        """

        x_t_shape = tf.shape(x_t)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """디퓨전 모델의 샘플입니다.

        Args:
            pred_noise: 디퓨전 모델에 의해 예측된 노이즈
            x: 노이즈가 예측된 주어진 타임스텝의 샘플
            t: 현재 타임스텝
            clip_denoised (bool): 예측된 노이즈를 지정된 범위 내에서 클리핑할지 여부
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        # t == 0일 때는 노이즈가 없습니다.
        nonzero_mask = tf.reshape(
            1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1]
        )
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
```

## 네트워크 아키텍처 {#network-architecture}

원래 시맨틱 세그멘테이션을 위해 개발된 U-Net은,
디퓨전 모델을 구현하는 데 널리 사용되는 아키텍처이지만, 약간의 수정이 가해졌습니다.

1. 네트워크는 두 가지 입력을 허용합니다. 이미지와 시간 단계
2. 특정 해상도(논문에서는 16x16)에 도달하면, 컨볼루션 블록 간의 셀프 어텐션
3. 가중치 정규화 대신 그룹 정규화

우리는 원래 논문에서 사용된 대부분의 것을 구현합니다.
우리는 네트워크 전체에서 `swish` 활성화 함수를 사용합니다.
우리는 분산 스케일링 커널 초기화자(variance scaling kernel initializer)를 사용합니다.

여기서 유일한 차이점은 `GroupNormalization` 레이어에 사용된 그룹 수입니다.
꽃 데이터 세트의 경우, 우리는 `groups=8` 값이 기본값인 `groups=32`보다 더 나은 결과를 생성한다는 것을 발견했습니다.
드롭아웃은 선택 사항이며, 과적합 가능성이 높은 곳에서 사용해야 합니다.
이 논문에서, 저자는 CIFAR10에 대해 트레이닝 할 때만 드롭아웃을 사용했습니다.

```python
# 사용할 커널 이니셜라이저
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    """셀프 어텐션을 적용합니다.

    Args:
        units: dense 레이어의 유닛 수
        groups: GroupNormalization 레이어에 사용할 그룹 수
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[
            :, None, None, :
        ]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply


def build_model(
    img_size,
    img_channels,
    widths,
    has_attention,
    num_res_blocks=2,
    norm_groups=8,
    interpolation="nearest",
    activation_fn=keras.activations.swish,
):
    image_input = layers.Input(
        shape=(img_size, img_size, img_channels), name="image_input"
    )
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")

    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)

    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)

    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)(
        [x, temb]
    )

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn
            )([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End 블록
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(3, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))(x)
    return keras.Model([image_input, time_input], x, name="unet")
```

## 트레이닝 {#training}

우리는 논문에서 설명한 것과 동일한 설정을 따라 디퓨전 모델을 트레이닝합니다.
우리는 `2e-4`의 학습률을 가진 `Adam` 옵티마이저를 사용합니다.
우리는 0.999의 감쇠 계수를 가진 모델 매개변수에 EMA를 사용합니다.
우리는 모델을 노이즈 예측 네트워크로 취급합니다.
즉, 모든 트레이닝 단계에서,
우리는 이미지 배치와 해당 시간 단계를 UNet에 입력하고,
네트워크는 노이즈를 예측으로 출력합니다.

유일한 차이점은 트레이닝 중에 생성된 샘플의 품질을 평가하기 위해,
커널 인셉션 거리(KID, Kernel Inception Distance) 또는 프레셰 인셉션 거리(FID, Frechet Inception Distance)를 사용하지 않는다는 것입니다.
이는 이 두 가지 지표가 모두 계산이 많기 때문인데, 구현의 간결성을 위해 스킵합니다.

**참고:** 논문과 일치하고 이론적으로 타당한 평균 제곱 오차를 손실 함수로 사용하고 있습니다.
그러나, 실제로는, 평균 절대 오차 또는 Huber 손실을 손실 함수로 사용하는 것도 일반적입니다.

```python
class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema

    def train_step(self, images):
        # 1. 배치 크기를 가져옵니다.
        batch_size = tf.shape(images)[0]

        # 2. 시간 단계를 uniform 샘플링
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        with tf.GradientTape() as tape:
            # 3. 배치의 이미지에 추가할 샘플 랜덤 노이즈
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. 노이즈로 이미지를 디퓨전시킵니다.
            images_t = self.gdf_util.q_sample(images, t, noise)

            # 5. 디퓨전된 이미지와 시간 단계를 네트워크에 전달합니다.
            pred_noise = self.network([images_t, t], training=True)

            # 6. 손실을 계산합니다.
            loss = self.loss(noise, pred_noise)

        # 7. 그래디언트를 얻습니다.
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. 네트워크의 가중치를 업데이트합니다
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. EMA 가중치를 사용하여 네트워크의 가중치 값을 업데이트합니다.
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. 손실 값을 반환합니다.
        return {"loss": loss}

    def generate_images(self, num_images=16):
        # 1. 무작위로 노이즈를 샘플링합니다. (역방향 프로세스의 시작점)
        samples = tf.random.normal(
            shape=(num_images, img_size, img_size, img_channels), dtype=tf.float32
        )
        # 2. 모델에서 반복적으로 샘플링합니다.
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
        # 3. 생성된 샘플을 반환합니다.
        return samples

    def plot_images(
        self, epoch=None, logs=None, num_rows=2, num_cols=8, figsize=(12, 5)
    ):
        """Utility to plot images using the diffusion model during training."""
        generated_samples = self.generate_images(num_images=num_rows * num_cols)
        generated_samples = (
            tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0)
            .numpy()
            .astype(np.uint8)
        )

        _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i, image in enumerate(generated_samples):
            if num_rows == 1:
                ax[i].imshow(image)
                ax[i].axis("off")
            else:
                ax[i // num_cols, i % num_cols].imshow(image)
                ax[i // num_cols, i % num_cols].axis("off")

        plt.tight_layout()
        plt.show()


# unet 모델 빌드
network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)
ema_network = build_model(
    img_size=img_size,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    activation_fn=keras.activations.swish,
)
ema_network.set_weights(network.get_weights())  # 처음에는 가중치가 동일합니다.

# Gaussian 디퓨전 유틸리티의 인스턴스를 가져옵니다.
gdf_util = GaussianDiffusion(timesteps=total_timesteps)

# 모델을 얻습니다.
model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)

# 모델을 컴파일합니다.
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
)

# 모델을 트레이닝합니다.
model.fit(
    train_ds,
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)],
)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
31/31 [==============================] - ETA: 0s - loss: 0.7746
```

![png](/images/examples/generative/ddpm/ddpm_13_1.png)

```plain
31/31 [==============================] - 194s 4s/step - loss: 0.7668

<keras.callbacks.History at 0x7fc9e86ce610>
```

{{% /details %}}

## 결과 {#results}

우리는 이 모델을 V100 GPU에서 800 에포크 동안 트레이닝시켰고,
각 에포크는 완료하는 데 거의 8초가 걸렸습니다.
우리는 여기에 그 가중치를 로드하고,
순수한 노이즈에서 시작하여 몇 개의 샘플을 생성합니다.

```python
!curl -LO https://github.com/AakashKumarNain/ddpms/releases/download/v3.0.0/checkpoints.zip
!unzip -qq checkpoints.zip
```

```python
# 모델 가중치 로드
model.ema_network.load_weights("checkpoints/diffusion_model_checkpoint")

# 샘플을 생성하고 플롯합니다.
model.plot_images(num_rows=4, num_cols=8)
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  222M  100  222M    0     0  16.0M      0  0:00:13  0:00:13 --:--:-- 14.7M
```

{{% /details %}}

![png](/images/examples/generative/ddpm/ddpm_16_0.png)

## 결론 {#conclusion}

우리는 DDPM 논문의 저자가 구현한 것과 정확히 같은 방식으로 디퓨전 모델을 성공적으로 구현하고 트레이닝했습니다.
원본 구현은 [여기](https://github.com/hojonathanho/diffusion)에서 찾을 수 있습니다.

모델을 개선하기 위해 시도할 수 있는 몇 가지 사항이 있습니다.

1. 각 블록의 너비를 늘립니다.

   - 더 큰 모델은 더 적은 에포크로 노이즈를 제거하는 법을 배울 수 있지만,
   - 과적합을 처리해야 할 수도 있습니다.

2. 분산 스케줄링을 위해 선형 스케줄을 구현했습니다.
   - 코사인 스케줄링과 같은 다른 방식을 구현하고 성능을 비교할 수 있습니다.

## 참조 {#references}

1.  [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
2.  [Author's implementation](https://github.com/hojonathanho/diffusion)
3.  [A deep dive into DDPMs](https://magic-with-latents.github.io/latent/posts/ddpms/part3/)
4.  [Denoising Diffusion Implicit Models]({{< relref "/docs/examples/generative/ddim" >}})
5.  [Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
6.  [AIAIART](https://www.youtube.com/watch?v=XTs7M6TSK9I&t=14s)
