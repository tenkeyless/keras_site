---
slug: getting_started
title: Keras로 시작하기
linkTitle: 시작하기
toc: true
weight: 1
type: docs
---

{{< original checkedAt="2024-03-28" >}}

## 학습 리소스 {#learning-resources}

Keras 소개 한 페이지짜리 가이드를 찾고 계신 머신러닝 엔지니어이신가요? [엔지니어를 위한 Keras 소개]({{< relref "/docs/getting_started/intro_to_keras_for_engineers.md" >}}) 가이드를 읽어보세요.

Keras 3와 그 기능에 대해 더 자세히 알고 싶으신가요? [Keras 3 출시 발표]({% link index.md %})를 참조하세요.

Keras API의 다양한 부분에 대한 심층적인 사용법을 다루는 자세한 가이드를 찾고 계신가요? [Keras 개발자 가이드]({% link docs/04-guides/00-guides.md %})를 읽어보세요.

다양한 사용 사례에서 Keras가 실제로 작동하는 모습을 보여주는 튜토리얼을 찾고 계신가요? 컴퓨터 비전, 자연어 처리, 생성 AI 분야에서 Keras 모범 사례를 보여주는 150개 이상의 잘 설명된 노트북인 [Keras 코드 예제]({% link docs/06-examples/00-examples.md %})를 참조하세요.

---

## 케라스 3 설치하기

{: #installing-keras-3}

<!-- ## Installing Keras 3 -->

PyPI에서 다음을 통해 Keras를 설치할 수 있습니다:

```shell
pip install --upgrade keras
```

다음을 통해 로컬 Keras 버전 번호를 확인할 수 있습니다:

```python
import keras
print(keras.__version__)
```

Keras 3를 사용하려면 백엔드 프레임워크(JAX, TensorFlow 또는 PyTorch)도 설치해야 합니다:

- [JAX 설치하기](https://jax.readthedocs.io/en/latest/installation.html)
- [TensorFlow 설치](https://www.tensorflow.org/install)
- [PyTorch 설치](https://pytorch.org/get-started/locally/)

텐서플로우 2.15를 설치한 경우, 이후 케라스 3를 재설치해야 합니다. 그 이유는 `tensorflow==2.15`가 Keras 설치를 `keras==2.15`로 덮어쓰기 때문입니다. 텐서플로우 2.16 버전부터는 기본적으로 케라스 3이 설치되므로 이 단계는 필요하지 않습니다.

### KerasCV 및 KerasNLP 설치하기

{: #installing-kerascv-and-kerasnlp}

<!-- ### Installing KerasCV and KerasNLP -->

KerasCV와 KerasNLP는 pip를 통해 설치할 수 있습니다:

```shell
pip install --upgrade keras-cv
pip install --upgrade keras-nlp
pip install --upgrade keras
```

---

## 백엔드 구성하기

{: #configuring-your-backend}

<!-- ## Configuring your backend -->

환경 변수 `KERAS_BACKEND`를 내보내거나 `~/.keras/keras.json`에서 로컬 구성 파일을 편집하여 백엔드를 구성할 수 있습니다. 사용 가능한 백엔드 옵션은 다음과 같습니다: `"jax"`, `"tensorflow"`, `"torch"`. 예시:

```shell
export KERAS_BACKEND="jax"
```

Colab에서는 다음과 같이 할 수 있습니다:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
```

**참고:** Keras를 import 하기 전에 백엔드를 구성해야 하며, 패키지를 import 한 후에는 백엔드를 변경할 수 없습니다.

### GPU 종속성

{: #gpu-dependencies}

<!-- ### GPU dependencies -->

#### Colab 또는 Kaggle

{: #colab-or-kaggle}

<!-- #### Colab or Kaggle -->

Colab 또는 Kaggle에서 실행하는 경우, GPU가 올바른 CUDA 버전으로 이미 구성되어 있어야 합니다. 일반적으로 Colab 또는 Kaggle에 최신 버전의 CUDA를 설치하는 것은 불가능합니다. pip 인스톨러가 있긴 하지만, 사전 설치된 NVIDIA 드라이버에 의존하며 Colab 또는 Kaggle에서 드라이버를 업데이트할 수 있는 방법이 없습니다.

#### 범용 GPU 환경

{: #universal-gpu-environment}

<!-- #### Universal GPU environment -->

모든 백엔드에서 GPU를 사용할 수 있는 "범용 환경"을 만들려면, [Colab에서 사용하는 종속성 버전](https://colab.sandbox.google.com/drive/13cpd3wCwEHpsmypY9o6XB6rXgBm5oSxu)을 따르는 것이 좋습니다. (이 문제를 정확히 해결하고자 함) 당신은 [여기](https://developer.nvidia.com/cuda-downloads)에서 CUDA 드라이버를 설치한 다음, 각각의 CUDA 설치 지침에 따라 백엔드를 pip 설치하면 됩니다: [JAX 설치](https://jax.readthedocs.io/en/latest/installation.html), [TensorFlow 설치](https://www.tensorflow.org/install), [PyTorch 설치](https://pytorch.org/get-started/locally/)를 참조하세요.

#### 가장 안정적인 GPU 환경

{: #most-stable-gpu-environment}

<!-- #### Most stable GPU environment -->

Keras 기여자이며 Keras 테스트를 실행하는 경우 이 설정을 권장합니다. 모든 백엔드를 설치하지만, 한 번에 하나의 백엔드에만 GPU 액세스 권한을 부여하여, 백엔드 간에 잠재적으로 충돌할 수 있는 종속성 요구 사항을 피합니다. 다음 백엔드별 요구 사항 파일을 사용할 수 있습니다:

- [requirements-jax-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-jax-cuda.txt)
- [requirements-tensorflow-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-tensorflow-cuda.txt)
- [requirements-torch-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-torch-cuda.txt)

이들은 pip를 통해 모든 CUDA 지원 종속 요소를 설치합니다. NVIDIA 드라이버가 사전 설치되어 있을 것으로 예상합니다. CUDA 버전 불일치를 방지하기 위해 각 백엔드에 대해 깨끗한 파이썬 환경을 권장합니다. 예를 들어, 다음은 [Conda](https://docs.conda.io/en/latest/)를 사용하여 JAX GPU 환경을 만드는 방법입니다:

```shell
conda create -y -n keras-jax python=3.10
conda activate keras-jax
pip install -r requirements-jax-cuda.txt
pip install --upgrade keras
```

---

## TensorFlow + Keras 2 이전 버전과의 호환성

{: #tensorflow--keras-2-backwards-compatibility}

<!-- ## TensorFlow + Keras 2 backwards compatibility -->

TensorFlow 2.0부터 TensorFlow 2.15(포함)까지, `pip install tensorflow`를 실행하면 해당 버전의 Keras 2도 설치됩니다. (예를 들어, `pip install tensorflow==2.14.0`은 `keras==2.14.0`을 설치합니다) 그런 다음, 해당 버전의 Keras는 `import keras`와 `from tensorflow import keras`를 통해 사용할 수 있습니다. ([`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) 네임스페이스)

TensorFlow 2.16부터는 `pip install tensorflow`를 실행하면 Keras 3가 설치됩니다. TensorFlow >= 2.16과 Keras 3를 사용하는 경우, 기본적으로 `from tensorflow import keras`([`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras))가 Keras 3가 됩니다.

한편, 레거시 Keras 2 패키지는 여전히 정기적으로 릴리스되고 있으며 PyPI에서 `tf_keras`(또는 `tf-keras`와 동일 - PyPI 패키지 이름에서 `-`와 `_`는 동일합니다)로 사용할 수 있습니다. 이를 사용하려면, `pip install tf_keras`를 통해 설치한 다음 `import tf_keras as keras`를 통해 import 할 수 있습니다.

TensorFlow 2.16 이상으로 업그레이드한 후에도, [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras)를 Keras 2에서 계속 사용하려면, [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras)가 `tf_keras`를 가리키도록 TensorFlow 설치를 구성할 수 있습니다. 이렇게 하려면:

1.  `tf_keras`를 설치합니다. TensorFlow는 기본적으로 설치하지 않습니다.
2.  환경 변수 `TF_USE_LEGACY_KERAS=1`을 내보냅니다.

환경 변수를 내보내는(export) 방법은 여러 가지가 있습니다:

1.  파이썬 인터프리터를 시작하기 전에 셸 명령어 `export TF_USE_LEGACY_KERAS=1`을 실행하면 됩니다.
2.  `.bashrc` 파일에 `export TF_USE_LEGACY_KERAS=1`을 추가할 수 있습니다. 이렇게 하면 셸을 다시 시작할 때 변수가 계속 내보내집니다.
3.  파이썬 스크립트로 시작할 수 있습니다:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

이 줄은 `import tensorflow` 문 앞에 와야 합니다.

---

## 호환성 매트릭스

{: #compatibility-matrix}

<!-- ## Compatibility matrix -->

### JAX 호환성

{: #jax-compatibility}

<!-- ### JAX compatibility -->

다음 Keras + JAX 버전은 서로 호환됩니다:

- `jax==0.4.20` & `keras~=3.0`

### TensorFlow 호환성

{: #tensorflow-compatibility}

<!-- ### TensorFlow compatibility -->

다음 케라스 + TensorFlow 버전은 서로 호환됩니다:

Keras 2를 사용하려면,

- `tensorflow~=2.13.0` & `keras~=2.13.0`
- `tensorflow~=2.14.0` & `keras~=2.14.0`
- `tensorflow~=2.15.0` & `keras~=2.15.0`

Keras 3을 사용하려면,

- `tensorflow~=2.16.1` & `keras~=3.0`

### PyTorch 호환성

{: #pytorch-compatibility}

<!-- ### PyTorch compatibility -->

다음 Keras + PyTorch 버전은 서로 호환됩니다:

- `torch~=2.1.0` & `keras~=3.0`
