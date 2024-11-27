---
title: Keras 에코시스템
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-18" >}}

Keras 프로젝트는 신경망을 구축하고 트레이닝하기 위한 core Keras API에 국한되지 않습니다.
머신러닝 워크플로의 모든 단계를 포괄하는 광범위한 관련 이니셔티브를 포괄합니다.

## KerasTuner {#kerastuner}

[KerasTuner 문서]({{< relref "/docs/keras_tuner" >}}) - [KerasTuner GitHub 저장소](https://github.com/keras-team/keras-tuner)

KerasTuner는 하이퍼파라미터 검색의 문제점을 해결하는,
사용하기 쉬운 확장 가능한 하이퍼파라미터 최적화 프레임워크입니다.
Define-by-run 구문으로 검색 공간을 쉽게 구성한 다음,
사용 가능한 검색 알고리즘 중 하나를 활용하여,
모델에 가장 적합한 하이퍼파라미터 값을 찾습니다.
KerasTuner에는 베이지안 최적화, 하이퍼밴드 및 랜덤 검색 알고리즘이 기본 제공되며,
연구자가 새로운 검색 알고리즘을 실험하기 위해 쉽게 확장할 수 있도록 설계되었습니다.

## KerasHub {#kerashub}

[KerasHub 문서]({{< relref "/docs/keras_hub" >}}) - [KerasHub GitHub 저장소](https://github.com/keras-team/keras-hub)

KerasHub는 전체 개발 주기 동안 사용자를 지원하는 자연어 처리 라이브러리입니다.
우리의 워크플로는 즉시 사용할 때 최첨단 사전 설정 가중치와 아키텍처를 갖춘 모듈식 구성 요소로 구축되며,
더 많은 제어가 필요할 때 쉽게 커스터마이즈 할 수 있습니다.

## KerasCV {#kerascv}

[KerasCV 문서]({{< relref "/docs/keras_cv" >}}) - [KerasCV GitHub 저장소](https://github.com/keras-team/keras-cv)

KerasCV는 응용 컴퓨터 비전 엔지니어가,
이미지 분류, 객체 감지, 이미지 분할, 이미지 데이터 보강 등과 같은 일반적인 사용 사례에 대한,
프로덕션 등급의 최첨단 트레이닝 및 추론 파이프라인을 빠르게 조립하는 데 활용할 수 있는,
모듈식 빌딩 블록(레이어, 메트릭, 손실, 데이터 보강)의 저장소입니다.

KerasCV는 Keras API의 수평적 확장으로 이해할 수 있습니다.
구성 요소는 core Keras에 추가하기에는 너무 특수화된 새로운 퍼스트 파티 Keras 객체(레이어, 메트릭 등)이지만,
나머지 Keras API와 동일한 수준의 세련미와 이전 버전과의 호환성이 보장됩니다.

## AutoKeras {#autokeras}

[AutoKeras 문서](https://autokeras.com/) - [AutoKeras GitHub 저장소](https://github.com/keras-team/autokeras)

AutoKeras는 Keras를 기반으로 하는 AutoML 시스템입니다.
Texas A&M University의 [DATA Lab](http://faculty.cs.tamu.edu/xiahu/index.html)에서 개발했습니다.
AutoKeras의 목표는 모든 사람이 머신러닝을 사용할 수 있도록 하는 것입니다.
몇 줄로 머신 러닝 문제를 해결할 수 있는 [`ImageClassifier`](https://autokeras.com/tutorial/image_classification/) 또는 [`TextClassifier`](https://autokeras.com/tutorial/text_classification/)와 같은 높은 레벨 엔드투엔드 API를 제공하며,
아키텍처 검색을 수행하는 [유연한 빌딩 블록](https://autokeras.com/tutorial/customized/)도 제공합니다.

```python
import autokeras as ak

clf = ak.ImageClassifier()
clf.fit(x_train, y_train)
results = clf.predict(x_test)
```
