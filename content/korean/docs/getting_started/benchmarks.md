---
slug: benchmarks
title: Keras 3 벤치마크
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

우리는 TensorFlow를 사용하는 Keras 2와 함께,
Keras 3의 세 백엔드([TensorFlow](https://tensorflow.org/),
[JAX](https://jax.readthedocs.io/en/latest/),
[PyTorch](https://pytorch.org/))를 벤치마크했습니다.
결과를 재생성하기 위한 코드와 설정 세부 정보는
[여기](https://github.com/haifeng-jin/keras-benchmarks/tree/v0.0.5)에서 찾을 수 있습니다.

## 모델 {#models}

우리는 생성 및 비생성 AI 작업 모두에 인기 있는 컴퓨터 비전 및 자연어 처리 모델 세트를 선택했습니다.
우리의 선택은 아래 표를 참조하세요.

**Table 1**: 벤치마킹에 사용되는 모델

|     | 비생성형            | 생성형                 |
| --- | ------------------- | ---------------------- |
| CV  | SegmentAnything[^1] | StableDiffusion[^2]    |
| NLP | BERT[^3]            | Gemma[^4], Mistral[^5] |

우리는 각 프레임워크에서 달성할 수 있는 최상의 성능을 측정하는 것이 아니라,
일반적인 사용자 워크플로의 기본 제공 성능을 측정합니다.
이 목표를 염두에 두고,
우리는 모델의 Keras 버전에 대해 KerasCV와 KerasNLP의 기존 구현을 활용했습니다.

## 하드웨어 {#hardware}

모든 벤치마크는 12개 vCPU와 85GB 호스트 메모리를 갖춘,
머신 타입이 `a2-highgpu-1g`인 Google Cloud Compute Engine에서,
40GB의 GPU 메모리를 갖춘 단일 NVIDIA A100 GPU로 수행되었습니다.

## 결과 {#results}

Table 2는 스텝당 밀리초 단위로 벤치마킹 결과를 표시합니다.
각 스텝은 단일 데이터 배치에 대한 트레이닝 또는 예측을 포함합니다.
결과는 모델 생성 및 컴파일 오버헤드를 포함하는 첫 번째 단계를 제외한, 100개 스텝에 대한 평균입니다.

공정한 비교를 위해,
동일한 모델 및 작업(fit 또는 predict)인 경우, 프레임워크 전체에서 동일한 배치 크기를 사용합니다.
그러나 다른 모델 및 작업의 경우, 크기와 아키텍처가 다르기 때문에,
메모리 부족(너무 큼)이나 GPU 사용률 저하(너무 작음)를 방지하기 위해 다른 배치 크기를 사용합니다.

대규모 언어 모델(Gemma 및 Mistral)의 경우에도,
유사한 수의 매개변수(7B)를 가진 동일한 모델 타입이므로 동일한 배치 크기를 사용했습니다.
또한 사용자가 널리 요청하기 때문에, 배치 크기가 1인 텍스트 생성을 벤치마킹했습니다.
트레이닝 및 추론에는 `bfloat16` 정밀도를 사용했고,
트레이닝(파인 튜닝)에는 LoRA6[^6]를 사용했습니다.

기본 성능을 측정하기 위해, 모든 기본 설정을 사용하려고 합니다.
예를 들어, 가능한 한 구성이 적은 높은 레벨 API(예: Keras `model.fit()` 사용)를 사용합니다.

이는 특정 하드웨어/프레임워크/모델 조합에 대한 최적화된 구현을 측정하는 것과는 상당히 다릅니다.
다양한 프레임워크에 대한 최상의 최적화된 결과는
[MLPerf](https://mlcommons.org/benchmarks/)를 참조하세요.

**Table 2**: 벤치마킹 결과. 속도는 ms/스텝으로 측정됩니다. 낮을수록 좋습니다.

|                                | 배치 크기 | Keras 2 (TensorFlow) | Keras 3 (TensorFlow) | Keras 3 (JAX) | Keras 3 (PyTorch) (eager) | Keras 3 (베스트) |
| ------------------------------ | --------- | -------------------- | -------------------- | ------------- | ------------------------- | ---------------- |
| **SegmentAnything (fit)**      | 1         | 386.93               | **355.25**           | 361.69        | 1,388.87                  | **355.25**       |
| **SegmentAnything (predict)**  | 4         | 1,859.27             | 438.50               | **376.34**    | 1,720.96                  | **376.34**       |
| **Stable Diffusion (fit)**     | 8         | 1,023.21             | 392.24               | **391.21**    | 823.44                    | **391.21**       |
| **Stable Diffusion (predict)** | 13        | 649.71               | **616.04**           | 627.27        | 1,337.17                  | **616.04**       |
| **BERT (fit)**                 | 32        | 486.00               | **214.49**           | 222.37        | 808.68                    | **214.49**       |
| **BERT (predict)**             | 256       | 470.12               | 466.01               | **418.72**    | 1,865.98                  | **418.72**       |
| **Gemma (fit)**                | 8         | NA                   | 232.52               | 273.67        | 525.15                    | **232.52**       |
| **Gemma (generate)**           | 32        | NA                   | 1,134.91             | **1,128.21**  | 7,952.67\*                | **1,128.21**     |
| **Gemma (generate)**           | 1         | NA                   | 758.57               | **703.46**    | 7,649.40\*                | **703.46**       |
| **Mistral (fit)**              | 8         | NA                   | **185.92**           | 213.22        | 452.12                    | **185.92**       |
| **Mistral (generate)**         | 32        | NA                   | 966.06               | **957.25**    | 10,932.59\*               | **957.25**       |
| **Mistral (generate)**         | 1         | NA                   | 743.28               | **679.30**    | 11,054.67\*               | **679.30**       |

\* _PyTorch 백엔드를 사용한 LLM 추론은 KerasHub가, HuggingFace와 달리, static 시퀀스 패딩을 사용하기 때문에, 현재 비정상적으로 느립니다. 이는 곧 해결될 것입니다._

## 토론 {#discussion}

### 주요 발견 1: "최고의" 백엔드는 없습니다. {#key-finding-1-there-is-no-best-backend}

Keras의 세 가지 백엔드는 각각 고유한 강점을 제공합니다.
중요한 점은, 성능 관점에서 다른 백엔드를 지속적으로 앞지르는 단일 백엔드는 없다는 것입니다.
가장 빠른 백엔드는 종종 특정 모델 아키텍처에 따라 달라집니다.

이는 최적의 성능을 추구할 때 프레임워크 선택성의 가치를 강조합니다.
Keras 3는 백엔드를 원활하게 전환하여, 모델에 이상적인 매치를 찾을 수 있도록 지원합니다.

### 주요 발견 2: Keras 3은 Keras 2보다 빠릅니다. {#key-finding-2-keras-3-is-faster-than-keras-2}

또한 Table 1에서 TensorFlow를 사용한 Keras 2에 비해
Keras 3(성능이 가장 좋은 백엔드 사용)의 처리량(스텝/ms) 증가를 계산했습니다.
결과는 다음 그림에 나와 있습니다.

![Figrue 2](/images/getting_started/benchmarks/jPncf0F.png "Figure 1: 처리량(스텝/ms) 측면에서 Keras 2보다 Keras 3의 속도 향상")

Keras 3는 모든 벤치마크 모델에서 Keras 2보다 지속적으로 성능이 우수했으며,
많은 케이스에서 상당한 속도 증가를 보였습니다.
SegmentAnything 추론은 380%의 놀라운 증가를 보였고,
StableDiffusion 학습 처리량은 150% 이상 증가했으며,
BERT 학습 처리량은 100% 이상 증가했습니다.

중요한 점은 Keras 3으로 업그레이드하고,
TensorFlow 백엔드를 계속 사용하더라도 여전히 성능이 향상된다는 것입니다.
이는 주로 Keras 2가 더 많은 TensorFlow fused ops을 직접 사용하기 때문이며,
이는 특정 사용 케이스에서 XLA 컴파일에 최적이 아닐 수 있습니다.

## 결론 {#conclusions}

프레임워크 성능은 특정 모델에 크게 좌우됩니다.
Keras 3는 작업에 가장 빠른 프레임워크를 선택할 수 있는 권한을 부여합니다.
이는 거의 항상 Keras 2보다 성능이 뛰어납니다.

[^1]: Kirillov, Alexander, et al. "Segment anything." ICCV (2023).
[^2]: Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." CVPR (2022).
[^3]: Kenton, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL (2019).
[^4]: Banks, Jeanine, et al. "Gemma: Introducing new state-of-the-art open models." The Keyword, Google (2024).
[^5]: Jiang, Albert Q., et al. "Mistral 7B." arXiv preprint arXiv:2310.06825 (2023).
[^6]: Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." ICLR (2022).
