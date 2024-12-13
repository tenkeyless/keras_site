---
title: PyTorch로 멀티 GPU 분산 트레이닝하기
linkTitle: PyTorch 분산 트레이닝
toc: true
weight: 17
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

**{{< t f_author >}}** [fchollet](https://twitter.com/fchollet)  
**{{< t f_date_created >}}** 2023/06/29  
**{{< t f_last_modified >}}** 2023/06/29  
**{{< t f_description >}}** PyTorch로 Keras 모델을 사용하여, 멀티 GPU 트레이닝을 진행하는 가이드.

{{< cards cols="2" >}}
{{< card link="https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/distributed_training_with_torch.ipynb" title="Colab" tag="Colab" tagType="warning">}}
{{< card link="https://github.com/keras-team/keras-io/blob/master/guides/distributed_training_with_torch.py" title="GitHub" tag="GitHub">}}
{{< /cards >}}

## 소개 {#introduction}

일반적으로 여러 디바이스에 계산을 분산시키는 방법에는 두 가지가 있습니다:

- **데이터 병렬 처리**
  - **데이터 병렬 처리**에서는 하나의 모델이 여러 장치나 여러 머신에 복제됩니다.
  - 각 장치는 서로 다른 배치의 데이터를 처리한 후, 결과를 병합합니다.
  - 이 설정에는 다양한 변형이 있으며, 서로 다른 모델 복제본이 결과를 병합하는 방식이나,
    각 배치마다 동기화되는지 여부 등에 차이가 있습니다.
- **모델 병렬 처리**
  - **모델 병렬 처리**에서는 하나의 모델의 다른 부분이 서로 다른 장치에서 실행되어, 하나의 데이터 배치를 함께 처리합니다.
  - 이는 여러 가지 브랜치를 특징으로 하는 자연스럽게 병렬화된 아키텍처를 가진 모델에 가장 적합합니다.

이 가이드는 데이터 병렬 처리, 특히 **동기식 데이터 병렬 처리**에 중점을 둡니다.
여기서 모델의 서로 다른 복제본은 각 배치를 처리한 후 동기화됩니다.
동기화는 모델의 수렴 동작을 단일 장치에서의 트레이닝과 동일하게 유지시킵니다.

특히, 이 가이드는 PyTorch의 `DistributedDataParallel` 모듈 래퍼를 사용하여,
Keras를 여러 GPU(일반적으로 2~16개)에서 트레이닝하는 방법을 가르칩니다.
이 설정은 단일 머신에 설치된 여러 GPU를 사용하는 싱글 호스트, 멀티 디바이스 트레이닝으로,
연구자들과 소규모 산업 워크플로우에서 가장 일반적으로 사용됩니다.

## 셋업 {#setup}

먼저, 우리가 트레이닝할 모델을 생성하는 함수와,
트레이닝할 데이터셋(MNIST)을 생성하는 함수를 정의해봅시다.

```python
import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np
import keras


def get_model():
    # 배치 정규화와 드롭아웃이 포함된, 간단한 컨브넷(convnet)을 만듭니다.
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(filters=12, kernel_size=3, padding="same", use_bias=False)(
        x
    )
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=24,
        kernel_size=6,
        use_bias=False,
        strides=2,
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=6,
        padding="same",
        strides=2,
        name="large_k",
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model


def get_dataset():
    # 데이터를 불러오고, 트레이닝과 테스트 세트로 나눕니다.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 이미지를 [0, 1] 범위로 스케일링합니다.
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # 이미지가 (28, 28, 1) shape을 가지도록 합니다.
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)

    # TensorDataset을 생성합니다.
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    return dataset
```

이제 GPU를 대상으로 하는 간단한 PyTorch 트레이닝 루프를 정의해보겠습니다. (`.cuda()` 호출에 주목하세요)

```python
def train_model(model, dataloader, num_epochs, optimizer, loss_fn):
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_loss_count = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # 순방향 패스 (Forward pass)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # 역방향 패스 및 최적화 (Backward and optimize)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_count += 1

        # 손실 통계 출력
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {running_loss / running_loss_count}"
        )
```

## 단일 호스트, 다중 장치 동기 트레이닝 {#single-host-multi-device-synchronous-training}

이 설정에서는, 여러 개의 GPU가 있는 하나의 머신(일반적으로 2~16개의 GPU)에서 트레이닝을 진행합니다.
각 디바이스는 **복제본(replica)**이라고 불리는 모델의 사본을 실행합니다.
간단히 설명하기 위해, 다음 내용에서는 8개의 GPU를 사용하는 것으로 가정하겠습니다. 이는 일반성을 잃지 않습니다.​

**작동 방식**

트레이닝의 각 단계에서:

- 현재 데이터 배치(**글로벌 배치**)는 8개의 서로 다른 하위 배치(**로컬 배치**)로 나뉩니다.
  예를 들어, 글로벌 배치에 512개의 샘플이 있으면, 8개의 로컬 배치 각각에는 64개의 샘플이 포함됩니다.
- 8개의 복제본 각각은 로컬 배치를 독립적으로 처리합니다:
  순전파를 실행한 후, 역전파를 수행하여, 모델 손실에 대한 가중치의 그래디언트를 출력합니다.
- 로컬 그래디언트로부터 발생한 가중치 업데이트는 8개의 복제본 간에 효율적으로 병합됩니다.
  이 병합은 각 스텝이 끝날 때 이루어지기 때문에, 복제본은 항상 동기화된 상태를 유지합니다.

실제로, 모델 레플리카의 가중치를 동기적으로 업데이트하는 과정은 각 개별 가중치 변수 레벨에서 처리됩니다.
이는 **미러드 변수(mirrored variable)** 객체를 통해 이루어집니다.

**사용 방법**

단일 호스트에서 여러 장치로 동기식 트레이닝을 수행하려면,
`torch.nn.parallel.DistributedDataParallel` 모듈 래퍼를 사용합니다. 아래는 그 동작 방식입니다:

- `torch.multiprocessing.start_processes`를 사용하여 장치별로 하나의 프로세스를 시작합니다.
  각 프로세스는 `per_device_launch_fn` 함수를 실행합니다.
- `per_device_launch_fn` 함수는 다음과 같은 작업을 수행합니다:
  - `torch.distributed.init_process_group`과 `torch.cuda.set_device`를 사용하여,
    해당 프로세스에서 사용할 장치를 설정합니다.
  - `torch.utils.data.distributed.DistributedSampler`와 `torch.utils.data.DataLoader`를 사용하여,
    데이터를 분산 데이터 로더로 변환합니다.
  - `torch.nn.parallel.DistributedDataParallel`을 사용하여,
    모델을 분산된 PyTorch 모듈로 변환합니다.
  - 그런 다음 `train_model` 함수를 호출합니다.
- `train_model` 함수는 각 프로세스에서 실행되며, 각 프로세스에서 모델은 별도의 장치를 사용합니다.

다음은 각 단계를 유틸리티 함수로 나눈 흐름입니다:

```python
# 설정
num_gpu = torch.cuda.device_count()
num_epochs = 2
batch_size = 64
print(f"Running on {num_gpu} GPUs")


def setup_device(current_gpu_index, num_gpus):
    # 장치 설정
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "56492"
    device = torch.device("cuda:{}".format(current_gpu_index))
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=num_gpus,
        rank=current_gpu_index,
    )
    torch.cuda.set_device(device)


def cleanup():
    torch.distributed.destroy_process_group()


def prepare_dataloader(dataset, current_gpu_index, num_gpus, batch_size):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_gpus,
        rank=current_gpu_index,
        shuffle=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader


def per_device_launch_fn(current_gpu_index, num_gpu):
    # 프로세스 그룹 설정
    setup_device(current_gpu_index, num_gpu)

    dataset = get_dataset()
    model = get_model()

    # 데이터 로더 준비
    dataloader = prepare_dataloader(dataset, current_gpu_index, num_gpu, batch_size)

    # torch 옵티마이저 생성
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # torch 손실 함수 생성
    loss_fn = torch.nn.CrossEntropyLoss()

    # 모델을 장치에 배치
    model = model.to(current_gpu_index)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[current_gpu_index], output_device=current_gpu_index
    )

    train_model(ddp_model, dataloader, num_epochs, optimizer, loss_fn)

    cleanup()
```

{{% details title="{{< t f_result >}}" closed="true" %}}

```plain
Running on 0 GPUs

/opt/conda/envs/keras-torch/lib/python3.10/site-packages/torch/cuda/__init__.py:611: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
```

{{% /details %}}

멀티 프로세스를 시작하는 시간입니다:

```python
if __name__ == "__main__":
    # notebooks을 지원하기 위해, "spawn" 대신 "fork" 방식을 사용합니다.
    torch.multiprocessing.start_processes(
        per_device_launch_fn,
        args=(num_gpu,),
        nprocs=num_gpu,
        join=True,
        start_method="fork",
    )
```

이제 끝났습니다!
