---
title: Keras FAQ
linkTitle: FAQ
toc: true
weight: 4
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

자주 묻는 질문 리스트입니다.

## 일반적인 질문 {#general-questions}

### 여러 (단일 머신의) GPU에서 Keras 모델을 어떻게 트레이닝 할 수 있나요? {#how-can-i-train-a-keras-model-on-multiple-gpus-on-a-single-machine}

여러 GPU에서 단일 모델을 실행하는 방법에는 **데이터 병렬 처리**와 **장치 병렬 처리**의 두 가지가 있습니다.
Keras는 두 가지를 모두 지원합니다.

데이터 병렬 처리의 경우, Keras는 JAX, TensorFlow 및 PyTorch의 빌트인 데이터 병렬 분산 API를 지원합니다.
다음 가이드를 참조하세요.

- [JAX로 멀티 GPU 분산 트레이닝]({{< relref "/docs/guides/distributed_training_with_jax" >}})
- [TensorFlow로 멀티 GPU 분산 트레이닝]({{< relref "/docs/guides/distributed_training_with_tensorflow" >}})
- [PyTorch로 멀티 GPU 분산 트레이닝]({{< relref "/docs/guides/distributed_training_with_torch" >}})

모델 병렬 처리의 경우, Keras는 자체 분산 API를 가지고 있으며, 현재는 JAX 백엔드에서만 지원합니다.
[`LayoutMap` API에 대한 문서]({{< relref "/docs/api/distribution" >}})를 참조하세요.

### TPU에서 Keras 모델을 어떻게 트레이닝 시킬 수 있나요? {#how-can-i-train-a-keras-model-on-tpu}

TPU는 Google Cloud에서 공개적으로 제공되는 딥러닝을 위한 빠르고 효율적인 하드웨어 가속기입니다.
Colab, Kaggle 노트북, GCP 딥러닝 VM을 통해 TPU를 사용할 수 있습니다.
(VM에서 `TPU_NAME` 환경 변수가 설정된 경우)

모든 Keras 백엔드(JAX, TensorFlow, PyTorch)는 TPU에서 지원되지만,
이 경우 JAX 또는 TensorFlow를 권장합니다.

**JAX 사용:**

TPU 런타임에 연결된 경우, 모델 구성 전에 이 코드 스니펫을 삽입하기만 하면 됩니다.

```python
import jax
distribution = keras.distribution.DataParallel(devices=jax.devices())
keras.distribution.set_distribution(distribution)
```

**TensorFlow 사용:**

TPU 런타임에 연결되면, `TPUClusterResolver`를 사용하여 TPU를 감지합니다.
그런 다음, `TPUStrategy`를 만들고 전략 범위(strategy scope)에서 모델을 구성합니다.

```python
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

with strategy.scope():
    # 여기에서 모델을 만들어보세요.
    ...
```

중요한 점은 다음과 같습니다.

- TPU 활용을 계속 유지할 수 있을 만큼, 빠르게 데이터를 읽을 수 있는지 확인하세요.
- TPU 활용을 계속 유지하기 위해, 그래프 실행당 여러 단계의 경사 하강을 실행하는 것을 고려하세요.
  `experimental_steps_per_execution` 인수 `compile()`를 통해 이를 수행할 수 있습니다.
  작은 모델의 경우, 상당한 속도 향상이 이루어집니다.

### Keras 구성 파일은 어디에 저장되나요? {#where-is-the-keras-configuration-file-stored}

모든 Keras 데이터가 저장되는 기본 디렉토리는 다음과 같습니다.

```plain
$HOME/.keras/
```

예를 들어, 저의 경우, MacBook Pro에서는 `/Users/fchollet/.keras/`입니다.

Windows 사용자는 `$HOME`을 `%USERPROFILE%`로 바꿔야 합니다.

Keras가 위의 디렉토리를 만들 수 없는 경우(예: 권한 문제로 인해), `/tmp/.keras/`가 백업으로 사용됩니다.

Keras 구성 파일은 `$HOME/.keras/keras.json`에 저장된 JSON 파일입니다.
기본 구성 파일은 다음과 같습니다.

```json
{
  "image_data_format": "channels_last",
  "epsilon": 1e-7,
  "floatx": "float32",
  "backend": "tensorflow"
}
```

다음 필드가 포함되어 있습니다.

- 이미지 처리 레이어 및 유틸리티에서 기본적으로 사용할 이미지 데이터 형식. (`channels_last` 또는 `channels_first`)
- 일부 작업에서 0으로 나누는 것을 방지하는데 사용할 `epsilon` 수치적 fuzz 계수.
- 기본 float 데이터 타입.
- 기본 백엔드. `"jax"`, `"tensorflow"`, `"torch"` 또는 `"numpy"` 중 하나일 수 있습니다.

마찬가지로, [`get_file()`]({{< relref "/docs/api/utils/#get_file" >}})로 다운로드한 것과 같은,
캐시된 데이터 세트 파일은 기본적으로 `$HOME/.keras/datasets/`에 저장되고,
Keras 애플리케이션의 캐시된 모델 가중치 파일은 기본적으로 `$HOME/.keras/models/`에 저장됩니다.

### Keras로 하이퍼파라미터 튜닝을 수행하는 방법은 무엇인가요? {#how-to-do-hyperparameter-tuning-with-keras}

[KerasTuner]({{< relref "/docs/keras_tuner" >}}) 사용을 권장합니다.

### 개발 중에 Keras를 사용하여, 재현 가능한 결과를 얻으려면 어떻게 해야 하나요? {#how-can-i-obtain-reproducible-results-using-keras-during-development}

고려해야 할 무작위성의 소스는 네 가지가 있습니다.

1. Keras 자체. (예: `keras.random` 연산 또는 `keras.layers`의 랜덤 레이어)
2. 현재 Keras 백엔드. (예: JAX, TensorFlow 또는 PyTorch)
3. Python 런타임.
4. CUDA 런타임. GPU에서 실행할 때, 일부 연산은 비결정적 출력을 갖습니다.
   이는 GPU가 많은 연산을 병렬로 실행하기 때문에, 실행 순서가 항상 보장되지 않기 때문입니다.
   부동 소수점의 제한된 정밀도로 인해, 여러 숫자를 더하더라도 더하는 순서에 따라 약간 다른 결과가 나올 수 있습니다.

Keras와 현재 백엔드 프레임워크를 모두 결정적으로 만들려면, 다음을 사용하세요.

```python
keras.utils.set_random_seed(1337)
```

Python을 결정론적으로 만들려면, 프로그램을 시작하기 전에 `PYTHONHASHSEED` 환경 변수를 `0`으로 설정해야 합니다.
(프로그램 자체 내에서가 아님)
이는 Python 3.2.3 이상에서 특정 해시 기반 작업
(예: 집합 또는 사전의 항목 순서, [Python 문서](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED) 참조)에 대해, 재현 가능한 동작을 하기 위해 필요합니다.

CUDA 런타임을 결정론적으로 만들려면,
TensorFlow 백엔드를 사용하는 경우,
[`tf.config.experimental.enable_op_determinism`](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism)을 호출합니다. 이렇게 하면 성능 비용이 발생합니다.
다른 백엔드에 대한 작업은 다를 수 있습니다. 백엔드 프레임워크의 문서를 직접 확인하세요.

### 모델을 저장하기 위한 옵션은 무엇입니까? {#what-are-my-options-for-saving-models}

_참고: Keras 모델을 저장하려면, pickle이나 cPickle을 사용하지 않는 것이 좋습니다._

**1) 전체 모델 저장(구성 + 가중치)**

전체 모델 저장은 다음을 포함하는 파일을 만드는 것을 의미합니다.

- 모델의 아키텍처, 모델 재생성을 허용합니다.
- 모델의 가중치
- 트레이닝 구성(손실, 최적화)
- 옵티마이저 상태, 중단한 지점에서 정확히 트레이닝을 재개할 수 있음.

전체 모델을 저장하는 기본적이고 권장되는 방법은, `model.save(your_file_path.keras)`를 실행하는 것입니다.

두 형식 중 하나로 모델을 저장한 후,
`model = keras.models.load_model(your_file_path.keras)`를 통해 다시 인스턴스화할 수 있습니다.

**예제:**

```python
from keras.saving import load_model

model.save('my_model.keras')
del model  # 기존 모델을 삭제합니다

# 이전 모델과 동일한 컴파일된 모델을 반환합니다.
model = load_model('my_model.keras')
```

**2) 가중치만 저장**

**모델의 가중치**를 저장해야 하는 경우,
아래 코드와 파일 확장자 `.weights.h5`를 사용하여, HDF5로 저장할 수 있습니다.

```python
model.save_weights('my_model.weights.h5')
```

모델을 인스턴스화하기 위한 코드가 있다고 가정하면,
저장한 가중치를 _동일한_ 아키텍처를 가진 모델에 로드할 수 있습니다.

```python
model.load_weights('my_model.weights.h5')
```

예를 들어 미세 조정이나 전이 학습을 위해,
공통 레이어가 있는 _다른_ 아키텍처에 가중치를 로드해야 하는 경우,
_레이어 이름_ 으로 로드할 수 있습니다.

```python
model.load_weights('my_model.weights.h5', by_name=True)
```

예제:

```python
"""
원본 모델이 다음과 같다고 가정합니다.

model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))
model.add(Dense(3, name='dense_2'))
...
model.save_weights(fname)
"""

# 새로운 모델
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 로드될 것입니다.
model.add(Dense(10, name='new_dense'))  # 로드되지 않을 것입니다.

# 첫 번째 모델에서 가중치를 로드합니다.
# 첫 번째 레이어인 dense_1에만 영향을 미칩니다.
model.load_weights(fname, by_name=True)
```

`h5py`를 설치하는 방법에 대한 지침은,
[모델을 저장하기 위해 HDF5 또는 h5py를 어떻게 설치할 수 있나요?](#how-can-i-install-hdf5-or-h5py-to-save-my-models)를 참조하세요.

**3) 구성만 저장(직렬화)**

**모델의 아키텍처**만 저장하고, 가중치나 트레이닝 구성은 저장하지 않는 경우, 다음을 수행할 수 있습니다.

```python
# JSON으로 저장
json_string = model.to_json()
```

생성된 JSON 파일은 사람이 읽을 수 있으며 필요한 경우 수동으로 편집할 수 있습니다.

그런 다음 이 데이터에서 새 모델을 빌드할 수 있습니다.

```python
# JSON으로부터 모델 재구성:
from keras.models import model_from_json
model = model_from_json(json_string)
```

**4) 저장된 모델에서 커스텀 레이어(또는 다른 커스텀 객체) 다루기**

로드하려는 모델에 커스텀 레이어 또는 다른 커스텀 클래스나 함수가 포함되어 있는 경우,
`custom_objects` 인수를 통해 로딩 메커니즘에 전달할 수 있습니다.

```python
from keras.models import load_model
# 모델에 "AttentionLayer" 클래스 인스턴스가 포함되어 있다고 가정합니다.
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

또는, [커스텀 객체 범위]({{< relref "/docs/api/models/model_saving_apis/serialization_utils/#customobjectscope-class" >}})를 사용할 수 있습니다.

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

커스텀 객체 처리 방식은 `load_model` 및 `model_from_json`에 대해서도 동일하게 작동합니다.

```python
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

### 모델을 저장하기 위해 HDF5 또는 h5py를 어떻게 설치합니까? {#how-can-i-install-hdf5-or-h5py-to-save-my-models}

Keras 모델을 HDF5 파일로 저장하기 위해, Keras는 h5py Python 패키지를 사용합니다.
이는 Keras의 의존성이며 기본적으로 설치되어야 합니다.
Debian 기반 배포판에서는 `libhdf5`를 추가로 설치해야 합니다.

```shell
sudo apt-get install libhdf5-serial-dev
```

h5py가 설치되었는지 확실하지 않으면, Python 셸을 열고 다음을 통해 모듈을 로드할 수 있습니다.

```python
import h5py
```

오류 없이 import한다면, 설치가 완료된 것입니다.
그렇지 않으면, [자세한 설치 지침을 여기](http://docs.h5py.org/en/latest/build.html)에서 확인할 수 있습니다.

### Keras를 어떻게 인용해야 하나요? {#how-should-i-cite-keras}

연구에 도움이 된다면 Keras를 출판물에 인용해 주세요. 다음은 BibTeX 항목의 예입니다.

```latex
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  howpublished={\url{https://keras.io}},
}
```

## 트레이닝 관련 질문 {#training-related-questions}

### "샘플", "배치", "에포크"는 무엇을 의미하나요? {#what-do-sample-batch-and-epoch-mean}

다음은 Keras `fit()`를 올바르게 활용하기 위해, 알아야 하고 이해해야 하는 몇 가지 일반적인 정의입니다.

- **샘플 (Sample)**: 데이터 세트의 한 요소입니다.
  예를 들어, 한 이미지는 컨볼루션 신경망의 **샘플**입니다.
  한 오디오 스니펫은 음성 인식 모델의 **샘플**입니다.

* **배치 (Batch)**: _N_ 개의 샘플 집합입니다.
  **배치**의 샘플은 병렬로 독립적으로 처리됩니다.
  트레이닝하는 경우, 배치는 모델에 대한 업데이트를 하나만 생성합니다.
  **배치**는 일반적으로 단일 입력보다 입력 데이터의 분포를 더 잘 근사합니다.
  배치가 클수록 근사치가 더 좋지만, 배치를 처리하는 데 시간이 더 오래 걸리고,
  업데이트가 하나만 생성되는 것도 사실입니다.
  추론(평가/예측)의 경우, 메모리를 초과하지 않는 하에 가능한 한 큰 배치 크기를 선택하는 것이 좋습니다.
  (배치가 클수록 일반적으로 평가/예측이 더 빨라짐)
* **에포크 (Epoch)**: 일반적으로 "전체 데이터 세트에 대한 한 번의 패스"로 정의되는 임의의 컷오프로,
  트레이닝을 여러 단계로 구분하는 데 사용되며, 로깅 및 주기적 평가에 유용합니다.
  Keras 모델의 `fit` 메서드와 함께 `validation_data` 또는 `validation_split`을 사용하는 경우,
  평가는 모든 **에포크**가 끝날 때마다 실행됩니다.
  Keras 내에서는 **에포크**가 끝날 때마다 실행되도록 특별히 설계된 [콜백]({{< relref "/docs/api/callbacks" >}})을 추가할 수 있습니다.
  이러한 예로는 학습률 변경 및 모델 체크포인팅(저장)이 있습니다.

### 트레이닝 손실이 테스트 손실보다 훨씬 높은 이유는 무엇입니까? {#why-is-my-training-loss-much-higher-than-my-testing-loss}

Keras 모델에는 트레이닝 모드와 테스트 모드의 두 가지 모드가 있습니다.
드롭아웃 및 L1/L2 가중치 정규화와 같은 정규화 메커니즘은 테스트 시에는 꺼집니다.
이는 트레이닝 시의 손실에는 반영되지만, 테스트 시의 손실에는 반영되지 않습니다.

게다가, Keras가 표시하는 트레이닝 손실은 **현재 에포크**에 대한 각 트레이닝 데이터 배치의 손실 평균입니다.
모델이 시간이 지남에 따라 변경되므로,
에포크의 첫 번째 배치에 대한 손실은 일반적으로 마지막 배치에 대한 손실보다 높습니다.
이로 인해 에포크별 평균이 낮아질 수 있습니다.
반면, 에포크에 대한 테스트 손실은 에포크가 끝날 때의 모델을 사용하여 계산되므로, 손실이 낮아집니다.

### 트레이닝 실행이 프로그램 중단으로부터 복구될 수 있는지 어떻게 보장할 수 있습니까? {#how-can-i-ensure-my-training-run-can-recover-from-program-interruptions}

중단된 트레이닝 실행에서 언제든지 복구할 수 있는 기능(장애 허용성)을 보장하려면,
에포크 번호와 가중치를 포함하여 트레이닝 진행 상황을 정기적으로 디스크에 저장하고,
다음에 `Model.fit()`를 호출할 때 로드하는 [`keras.callbacks.BackupAndRestore`]({{< relref "/docs/api/callbacks/backup_and_restore#backupandrestore-class" >}}) 콜백을 사용해야 합니다.

```python
import numpy as np
import keras

class InterruptingCallback(keras.callbacks.Callback):
  """의도적으로 트레이닝을 방해하기 위해 도입하는 콜백입니다."""
  def on_epoch_end(self, epoch, log=None):
    if epoch == 15:
      raise RuntimeError('Interruption')

model = keras.Sequential([keras.layers.Dense(10)])
optimizer = keras.optimizers.SGD()
model.compile(optimizer, loss="mse")

x = np.random.random((24, 10))
y = np.random.random((24,))

backup_callback = keras.callbacks.experimental.BackupAndRestore(
    backup_dir='/tmp/backup')
try:
  model.fit(x, y, epochs=20, steps_per_epoch=5,
            callbacks=[backup_callback, InterruptingCallback()])
except RuntimeError:
  print('***Handling interruption***')
  # 이것은 중단된 에포크부터 계속됩니다.
  model.fit(x, y, epochs=20, steps_per_epoch=5,
            callbacks=[backup_callback])
```

자세한 내용은 [콜백 문서]({{< relref "/docs/api/callbacks" >}})에서 확인하세요.

### 검증 손실이 더 이상 감소하지 않을 때, 트레이닝을 중단하려면 어떻게 해야 합니까?{#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore}

`EarlyStopping` 콜백을 사용할 수 있습니다.

```python
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

자세한 내용은 [콜백 문서]({{< relref "/docs/api/callbacks" >}})에서 확인하세요.

### 레이어를 동결하고, 미세 조정하려면 어떻게 해야 합니까? {#how-can-i-freeze-layers-and-do-fine-tuning}

**`trainable` 속성 설정**

모든 레이어 및 모델에는 `layer.trainable` boolean 속성이 있습니다.

```console
>>> layer = Dense(3)
>>> layer.trainable
True
```

모든 레이어 및 모델에서, `trainable` 속성을 설정할 수 있습니다. (True 또는 False)
`False`로 설정하면, `layer.trainable_weights` 속성은 비어 있습니다.

```console
>>> layer = Dense(3)
>>> layer.build(input_shape=(None, 3)) # 레이어의 가중치를 생성합니다
>>> layer.trainable
True
>>> layer.trainable_weights
[<KerasVariable shape=(3, 3), dtype=float32, path=dense/kernel>, <KerasVariable shape=(3,), dtype=float32, path=dense/bias>]
>>> layer.trainable = False
>>> layer.trainable_weights
[]
```

레이어에 `trainable` 속성을 설정하면,
모든 자식 레이어(`self.layers`의 내용)에 재귀적으로 설정됩니다.

**1) `fit()`로 트레이닝할 때:**

`fit()`로 미세 조정을 수행하려면, 다음을 수행합니다.

- 베이스 모델을 인스턴스화하고, 사전 트레이닝된 가중치를 로드합니다.
- 해당 베이스 모델을 동결합니다.
- 맨 위에 트레이닝 가능한 레이어를 추가합니다.
- `compile()` 및 `fit()`를 호출합니다.

다음과 같이 합니다:

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # ResNet50Base를 동결합니다.

assert model.layers[0].trainable_weights == []  # ResNet50Base에는 트레이닝 가능한 가중치가 없습니다.
assert len(model.trainable_weights) == 2  # Dense 레이어의 바이어스와 커널만 있습니다.

model.compile(...)
model.fit(...)  # ResNet50Base를 제외하고 Dense를 트레이닝합니다.
```

Functional API 또는 모델 서브클래싱 API를 사용하여 유사한 워크플로를 따를 수 있습니다.
`trainable`의 값을 변경한 _후에_ `compile()`을 호출하여 변경 사항을 고려하세요.
`compile()`을 호출하면, 모델의 트레이닝 단계 상태가 동결됩니다.

**2) 커스텀 트레이닝 루프를 사용하는 경우:**

트레이닝 루프를 작성할 때는, `model.trainable_weights`의 일부인 가중치만 업데이트해야 합니다.
(모든 `model.weights`가 아님)
간단한 TensorFlow 예는 다음과 같습니다.

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # ResNet50Base를 동결합니다.

# 데이터 세트의 배치에 걸쳐 반복합니다.
for inputs, targets in dataset:
    # GradientTape를 엽니다.
    with tf.GradientTape() as tape:
        # 포워드 패스.
        predictions = model(inputs)
        # 이 배치에 대한 손실 값을 계산합니다.
        loss_value = loss_fn(targets, predictions)

    # *trainable* 가중치에 대한 손실의 기울기를 얻습니다.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # 모델의 가중치를 업데이트합니다.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

**`trainable`과 `compile()`의 상호작용**

모델에서 `compile()`을 호출하는 것은 해당 모델의 동작을 "동결"하는 것을 의미합니다.
즉, 모델이 컴파일될 때의 `trainable` 속성 값은,
`compile`이 다시 호출될 때까지,
해당 모델의 수명 내내 보존되어야 합니다.
따라서, `trainable`을 변경하는 경우, 모델에서 `compile()`을 다시 호출하여 변경 사항을 적용해야 합니다.

예를 들어, 두 모델 A와 B가 일부 레이어를 공유하고 다음과 같은 경우:

- 모델 A가 컴파일됩니다.
- 공유 레이어의 `trainable` 속성 값이 변경됩니다.
- 모델 B가 컴파일됩니다.

그러면, 모델 A와 B는 공유 레이어에 대해 서로 다른 `trainable` 값을 사용합니다.
이 메커니즘은 다음을 수행하는 대부분의 기존 GAN 구현에 필수적입니다.

```python
discriminator.compile(...)  # `discriminator`의 가중치는 `discriminator`가 트레이닝될 때 업데이트되어야 합니다.
discriminator.trainable = False
gan.compile(...)  # `discriminator`는 `gan`의 하위 모델이므로, `gan`이 트레이닝될 때 업데이트되어서는 안 됩니다.
```

### `call()`의 `training` 인수와 `trainable` 속성의 차이점은 무엇인가요? {#whats-the-difference-between-the-training-argument-in-call-and-the-trainable-attribute}

`training`은 호출을 추론 모드 또는 트레이닝 모드에서 실행할지 여부를 결정하는 `call`의 boolean 인수입니다.
예를 들어, 트레이닝 모드에서 `Dropout` 레이어는 랜덤 드롭아웃을 적용하고 출력을 재조정합니다.
추론 모드에서는 동일한 레이어가 아무 작업도 수행하지 않습니다. 예:

```python
y = Dropout(0.5)(x, training=True)  # 트레이닝 시 *및* 추론 시에 드롭아웃을 적용합니다.
```

`trainable`은 트레이닝 중 손실을 최소화하기 위해,
레이어의 트레이닝 가능한 가중치를 업데이트해야 하는지 결정하는 boolean 레이어 속성입니다.
`layer.trainable`이 `False`로 설정된 경우,
`layer.trainable_weights`는 항상 빈 리스트가 됩니다. 예:

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # ResNet50Base를 동결합니다.

assert model.layers[0].trainable_weights == []  # ResNet50Base에는 트레이닝 가능한 가중치가 없습니다.
assert len(model.trainable_weights) == 2  # Dense 레이어의 바이어스와 커널만 있습니다.

model.compile(...)
model.fit(...)  # ResNet50Base를 제외하고 Dense를 트레이닝합니다.
```

보시다시피, "추론 모드 vs 트레이닝 모드"와 "레이어 가중치 트레이닝 가능성"은 두 가지 매우 다른 개념입니다.

다음을 상상해 보세요. 역전파를 통해 트레이닝 중에 스케일링 요소가 학습되는 드롭아웃 레이어입니다.
`AutoScaleDropout`이라고 이름을 지정하겠습니다.
이 레이어는 동시에 트레이닝 가능한 상태와, 추론 및 트레이닝에서 다른 동작을 갖습니다.
`trainable` 속성과 `training` 호출 인수는 독립적이므로, 다음을 수행할 수 있습니다.

```python
layer = AutoScaleDropout(0.5)

# 트레이닝 시 *및* 추론 시에 드롭아웃을 적용하고,
# 트레이닝 중에 스케일링 factor를 학습합니다.
y = layer(x, training=True)

assert len(layer.trainable_weights) == 1
```

```python
# *동결된* 스케일링 factor로
# 트레이닝 시 *및* 추론 시에 드롭아웃을 적용합니다.

layer = AutoScaleDropout(0.5)
layer.trainable = False
y = layer(x, training=True)
```

**_`BatchNormalization` 레이어의 특수 케이스_**

`BatchNormalization` 레이어의 경우,
`bn.trainable = False`를 설정하면 `training` 호출 인수도 기본적으로 `False`로 설정되어,
레이어가 트레이닝하는 동안 상태를 업데이트하지 않습니다.

이 동작은 `BatchNormalization`에만 적용됩니다.
다른 모든 레이어의 경우, 가중치 트레이닝 가능성과 "추론 vs 트레이닝 모드"는 독립적으로 유지됩니다.

### `fit()`에서, 검증 분할은 어떻게 계산됩니까? {#in-fit-how-is-the-validation-split-computed}

`model.fit`에서 `validation_split` 인수를 예를 들어 0.1로 설정하면,
사용된 검증 데이터는 데이터의 _마지막 10%_ 가 됩니다.
0.25로 설정하면, 데이터의 마지막 25%가 됩니다.
검증 분할을 추출하기 전에는 데이터가 셔플되지 않으므로,
검증은 문자 그대로 전달한 입력의 샘플의 _마지막_ x%에 해당합니다.

동일한 검증 세트가 모든 에포크에 사용됩니다. (`fit`에 대한 동일한 호출 내에서)

`validation_split` 옵션은 데이터가 Numpy 배열로 전달된 경우에만 사용할 수 있습니다.
(인덱싱할 수 없는 [`tf.data.Datasets`](https://www.tensorflow.org/api_docs/python/tf/data/Datasets)는 아님)

### `fit()`에서, 트레이닝 중에 데이터가 셔플되나요? {#in-fit-is-the-data-shuffled-during-training}

데이터를 NumPy 배열로 전달하고,
`model.fit()`의 `shuffle` 인수가 `True`(이것이 디폴트입니다)로 설정된 경우,
트레이닝 데이터는 각 에포크에서 전역적으로 랜덤으로 셔플됩니다.

데이터를 [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) 객체로 전달하고,
`model.fit()`의 `shuffle` 인수가 `True`로 설정된 경우,
데이터 세트는 로컬로 셔플됩니다. (버퍼된 셔플링)

[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) 객체를 사용하는 경우,
버퍼 크기를 제어할 수 있도록, (예: `dataset = dataset.shuffle(buffer_size)`를 호출함으로써) 사전에 데이터를 셔플링하는 것이 좋습니다.

검증 데이터는 셔플되지 않습니다.

### `fit()`으로 트레이닝할 때 메트릭을 모니터링하는 데 권장되는 방법은 무엇입니까? {#whats-the-recommended-way-to-monitor-my-metrics-when-training-with-fit}

손실 값과 메트릭 값은 `fit()` 호출로 표시되는 기본 진행률 막대를 통해 보고됩니다.
그러나, 콘솔에서 ASCII 숫자를 변경하는 것을 응시하는 것은 최적의 메트릭 모니터링 경험이 아닙니다.
[TensorBoard](https://www.tensorflow.org/tensorboard)를 사용하는 것이 좋습니다.
이것은 트레이닝 및 검증 메트릭의 보기 좋은 그래프를 표시하며,
트레이닝 중에 정기적으로 업데이트되며 브라우저에서 액세스할 수 있습니다.

[`TensorBoard` 콜백]({{< relref "/docs/api/callbacks/tensorboard/" >}})을 통해,
`fit()`와 함께 TensorBoard를 사용할 수 있습니다.

### `fit()`의 기능을 커스터마이즈해야 하는 경우 어떻게 해야 하나요? {#what-if-i-need-to-customize-what-fit-does}

두 가지 옵션이 있습니다.

**1) `Model` 클래스를 서브클래싱하고, `train_step`(및 `test_step`) 메서드를 재정의합니다.**

커스텀 업데이트 규칙을 사용하지만,
콜백, 효율적인 단계 퓨징(efficient step fusing) 등과 같이 `fit()`에서 제공하는 기능을 활용하려는 경우,
이 옵션이 더 좋습니다.

이 패턴은 Functional API로 모델을 빌드하는 것을 방해하지 않습니다.
이 경우 `inputs` 및 `outputs`로 모델을 인스턴스화하기 위해 만든 클래스를 사용합니다.
Sequential 모델도 마찬가지입니다.
이 경우 [`keras.Sequential`]({{< relref "/docs/api/models/sequential#sequential-class" >}})을 서브클래싱하고,
[`keras.Model`]({{< relref "/docs/api/models/model#model-class" >}}) 대신 `train_step`을 재정의합니다.

다음 가이드를 참조하세요.

- [JAX에서 커스텀 트레이닝 단계 작성]({{< relref "/docs/guides/custom_train_step_in_jax" >}})
- [TensorFlow에서 커스텀 트레이닝 단계 작성]({{< relref "/docs/guides/custom_train_step_in_tensorflow" >}})
- [PyTorch에서 커스텀 트레이닝 단계 작성]({{< relref "/docs/guides/custom_train_step_in_torch" >}})

**2) 낮은 레벨 커스텀 트레이닝 루프 작성**

이것은 모든 세부 사항을 제어하고 싶은 경우 좋은 옵션입니다. 하지만 다소 장황할 수 있습니다.

다음 가이드를 참조하세요.

- [JAX에서 커스텀 트레이닝 루프 작성]({{< relref "/docs/guides/writing_a_custom_training_loop_in_jax" >}})
- [TensorFlow에서 커스텀 트레이닝 루프 작성]({{< relref "/docs/guides/writing_a_custom_training_loop_in_tensorflow" >}})
- [PyTorch에서 커스텀 트레이닝 루프 작성]({{< relref "/docs/guides/writing_a_custom_training_loop_in_torch" >}})

### `Model` 메서드 `predict()`와 `__call__()`의 차이점은 무엇입니까? {#whats-the-difference-between-model-methods-predict-and-call}

[파이썬으로 딥러닝하기, 2판](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras)에서 발췌한 내용으로 답해 보겠습니다.

> `y = model.predict(x)`와 `y = model(x)`(여기서 `x`는 입력 데이터의 배열)는 모두
> "`x`에 대해 모델을 실행하고, 출력 `y`를 검색합니다."라는 의미입니다.
> 하지만 정확히 같은 것은 아닙니다.
>
> `predict()`는 배치에서 데이터에 걸쳐 반복합니다.
> (사실, `predict(x, batch_size=64)`를 통해 배치 크기를 지정할 수 있음)
> 그리고 출력의 NumPy 값을 추출합니다.
> 도식적으로는 다음과 같습니다.
>
> ```python
> def predict(x):
>     y_batches = []
>     for x_batch in get_batches(x):
>         y_batch = model(x_batch).numpy()
>         y_batches.append(y_batch)
>     return np.concatenate(y_batches)
> ```
>
> 즉, `predict()` 호출은 매우 큰 배열로 확장될 수 있습니다.
> 반면, `model(x)`는 메모리 내에서 발생하며 확장되지 않습니다.
> 한편으로, `predict()`는 미분할 수 없습니다.
> `GradientTape` 범위에서 호출하는 경우, 기울기를 검색할 수 없습니다.
>
> 모델 호출의 기울기를 검색해야 할 때는 `model(x)`를 사용해야 하고,
> 출력 값만 필요한 경우에는 `predict()`를 사용해야 합니다.
> 즉, 낮은 레벨 기울기 하강 루프를 작성하는 중이 아니라면, 항상 `predict()`를 사용하세요. (지금처럼)

## 모델링 관련 질문 {#modeling-related-questions}

### 중간 레이어의 출력(특성 추출)은 어떻게 얻을 수 있나요? {#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction}

Functional API와 Sequential API에서,
레이어가 정확히 한 번 호출된 경우,
`layer.output`을 통해 출력을 검색하고, `layer.input`을 통해 입력을 검색할 수 있습니다.
이렇게 하면, 다음과 같이 특성 추출 모델을 빠르게 인스턴스화할 수 있습니다.

```python
import keras
from keras import layers

model = Sequential([
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.GlobalMaxPooling2D(),
    layers.Dense(10),
])
extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])
features = extractor(data)
```

당연히, `call`을 오버라이드하는 `Model`의 서브클래스인 모델에서는 불가능합니다.

다음은 또다른 예입니다. 특정 이름이 지정된 레이어의 출력을 반환하는 `Model`을 인스턴스화합니다.

```python
model = ...  # 원본 모델을 생성합니다.

layer_name = 'my_layer'
intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model(data)
```

### Keras에서 사전 트레이닝된 모델을 어떻게 사용할 수 있나요? {#how-can-i-use-pre-trained-models-in-keras}

[`keras.applications`에서 사용 가능한 모델]({{< relref "/docs/api/applications" >}}) 또는
[KerasCV]({{< relref "/docs/keras_cv" >}}) 및
[KerasHub]({{< relref "/docs/keras_hub" >}})에서 사용 가능한 모델을 활용할 수 있습니다.

### stateful RNN을 어떻게 사용할 수 있나요? {#how-can-i-use-stateful-rnns}

RNN을 stateful로 만든다는 것은,
각 배치의 샘플에 대한 상태가, 다음 배치의 샘플에 대한 초기 상태로 재사용된다는 것을 의미합니다.

따라서 stateful RNN을 사용할 때, 다음이 가정됩니다.

- 모든 배치에 동일한 수의 샘플이 있습니다.
- `x1`과 `x2`가 연속적인 샘플 배치인 경우, 모든 `i`에 대해, `x2[i]`는 `x1[i]`의 후속 시퀀스입니다.

RNN에서 statefulness를 사용하려면, 다음이 필요합니다.

- 모델의 첫 번째 레이어에 `batch_size` 인수를 전달하여, 사용하는 배치 크기를 명시적으로 지정합니다.
  예를 들어, 시간 단계당 16개의 특성이 있는 10개의 시간 단계 시퀀스의 32개 샘플 배치의 경우, `batch_size=32`입니다.
- RNN 레이어에서 `stateful=True`를 설정합니다.
- `fit()`를 호출할 때, `shuffle=False`를 지정합니다.

누적된 상태를 재설정하려면, 다음을 수행합니다.

- 모델의 모든 레이어 상태를 재설정하려면, `model.reset_states()`를 사용합니다.
- 특정 stateful RNN 레이어의 상태를 재설정하려면, `layer.reset_states()`를 사용합니다.

예:

```python
import keras
from keras import layers
import numpy as np

x = np.random.random((32, 21, 16))  # 이것은 (32, 21, 16) 모양의 입력 데이터입니다.
# 우리는 이것을 길이 10의 시퀀스로 우리 모델에 공급할 것입니다.

model = keras.Sequential()
model.add(layers.LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(layers.Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 우리는 처음 10개를 주어진 11번째 타임스텝을 예측하도록 네트워크를 트레이닝합니다.
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# 네트워크 상태가 변경되었습니다. 후속 시퀀스를 제공할 수 있습니다.
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# LSTM 레이어의 상태를 재설정해 보겠습니다.
model.reset_states()

# 이 경우 다른 방법은 다음과 같습니다.
model.layers[0].reset_states()
```

`predict`, `fit`, `train_on_batch` 등의 메서드는 _모두_ 모델의 stateful 레이어의 상태를 업데이트합니다.
이를 통해, stateful 트레이닝 뿐만 아니라, stateful 예측도 수행할 수 있습니다.
