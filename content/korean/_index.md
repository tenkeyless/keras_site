---
title: Keras
layout: hextra-home
---

<div class="hx-mt-6 hx-mb-6">
{{< hextra/hero-headline >}}
  Keras
{{< /hextra/hero-headline >}}
</div>

<div class="hx-mb-12">
{{< hextra/hero-subtitle >}}
  간단함. 유연함. 강력함.
{{< /hextra/hero-subtitle >}}
</div>

<div class="hx-w-full hx-gap-4" style="--hextra-cards-grid-cols: 4; display: grid; grid-template-columns: repeat(auto-fill, minmax(max(150px, calc((100% - 2rem* 2) / var(--hextra-cards-grid-cols))), 1fr)); padding: 0 4em;">
{{< main_button rellink="/docs/getting_started" title="시작하기" background="radial-gradient(ellipse at 50% 80%,rgba(142,53,74,.15),hsla(0,0%,100%,0))" >}}
{{< main_button rellink="/docs/api" title="API 문서"  >}}
{{< main_button rellink="/docs/guides" title="가이드"  >}}
{{< main_button rellink="/docs/examples" title="예제"  >}}
</div>

{{< main_large_text title="이제 JAX, TensorFlow, PyTorch에서 Keras를 사용할 수 있습니다!" rellink="/docs/keras_3" linkTitle="Keras 3.0 릴리스 공지 읽기" >}}

<div class="hx-w-full" style="padding: 0 2rem; ">
{{< cards cols="3" >}}
{{< main_card title="Maciej Kula" subtitle="스태프 소프트웨어 엔지니어" workAt="Google" words="\"Keras는 YouTube Discovery의 새로운 모델링 인프라의 핵심 구성 요소 중 하나입니다. 이는 YouTube 추천의 주요 표면 전반에 걸쳐 8개 팀에 명확하고 일관된 API와 모델링 아이디어를 표현하는 공통 방법을 제공합니다.\"" >}}
{{< main_card title="Yiming Chen" subtitle="시니어 소프트웨어 엔지니어" workAt="Waymo" words="\"Keras는 상당히 단순화된 API, 표준화된 인터페이스 및 동작, 쉽게 공유할 수 있는 모델 구축 구성 요소, 크게 향상된 디버깅 가능성의 이점을 통해 Waymo ML 실무자의 개발 워크플로를 엄청나게 단순화했습니다.\"" >}}
{{< main_card title="Matthew Carrigan" subtitle="머신러닝 엔지니어" workAt="Hugging Face" words="\"소프트웨어 라이브러리에 대해 말할 수 있는 가장 좋은 점은 선택한 추상화가 완전히 자연스러워서, 원하는 작업에 대한 생각과 이를 코딩하는 방법에 대한 생각 사이에 마찰이 전혀 없다는 것입니다. Keras를 사용하면 바로 이런 결과를 얻을 수 있습니다.\"" >}}
{{< main_card title="Aiden Arnold, PhD" subtitle="수석 데이터 과학자" workAt="Rune Labs" words="\"Keras를 사용하면 직관적이고 효율적인 방식으로 딥러닝 모델의 프로토타입을 제작하고 연구하고 배포할 수 있습니다. 함수형 API는 코드를 이해하기 쉽고 스타일리시하게 만들어 우리 팀의 과학자들 간에 효과적인 지식 전달을 가능하게 합니다.\"" >}}
{{< main_card title="Abheesht Sharma" subtitle="리서치 과학자" workAt="Amazon" words="\"Keras는 모든 사용자를 위한 무언가를 제공합니다: 학계를 위한 쉬운 사용자 정의; 업계에서 사용할 수 있는 즉시 사용 가능한 고성능 모델 및 파이프라인과 학생을 위한 읽기 쉬운 모듈식 코드를 제공합니다. Keras를 사용하면 낮은 레벨 세부 사항에 대해 걱정할 필요 없이, 실험을 빠르게 반복하는 것이 매우 간단합니다.\"" >}}
{{< main_card title="Santiago L. Valdarrama" subtitle="머신러닝 컨설턴트" words="\"Keras는 딥러닝 모델을 구축하고 운영하기 위한 완벽한 추상화 레이어입니다. 저는 2018년부터 이를 사용하여 세계 최대 규모의 일부 회사를 위한 모델을 개발하고 배포했습니다. [...] Keras, TensorFlow 및 TFX의 조합에는 경쟁자가 없습니다.\"" >}}
{{< main_card title="Margaret Maynard-Reid" subtitle="머신러닝 엔지니어" words="\"제가 개인적으로 Keras에서 가장 마음에 드는 점은(직관적인 API 외에도) 연구에서 생산으로의 전환이 쉽다는 것입니다. Keras 모델을 트레이닝하고, TF Lite로 변환한 후, 모바일 및 에지 기기에 배포할 수 있습니다.\"" >}}
{{< main_card title="Aakash Nain" subtitle="리서치 엔지니어" words="\"Keras는 연구의 유연성과 배포의 일관성을 얻을 수 있는 최적의 장소입니다. Keras와 딥러닝과의 관계는 Ubuntu와 운영체제와의 관계처럼 보입니다.\"" >}}
{{< main_card title="Gareth Collins" subtitle="머신러닝 엔지니어" words="\"Keras의 사용자 친화적인 설계는 배우기 쉽고 사용하기 쉽다는 것을 의미합니다. [...] 다양한 플랫폼에서 모델을 빠르게 프로토타이핑하고 배포할 수 있습니다.\"" >}}
{{< /cards >}}
</div>

{{< cards cols="1">}}
{{< main_large_card title="개발자를 위한 초능력." image="/images/showcase-superpower.png" subtitle="Keras의 목적은 머신러닝 기반 앱을 출시하려는 모든 개발자에게 불공평한 이점을 제공하는 것입니다. Keras는 디버깅 속도, 코드 우아함 및 간결성, 유지 관리 가능성, 배포 가능성에 중점을 둡니다. Keras를 선택하면 코드베이스가 더 작고, 읽기 쉽고, 반복하기가 더 쉽습니다. XLA 컴파일 덕분에 JAX 및 TensorFlow를 사용한 모델이 더 빠르게 실행되고, TF Serving, TorchServe, TF Lite, TF.js 등과 같은 TensorFlow 및 PyTorch 생태계의 서비스 구성 요소 덕분에 모든 표면(서버, 모바일, 브라우저, 임베디드)에 배포하기가 더 쉽습니다." >}}
{{< main_large_card title="인간을 위한 딥러닝." image="/images/showcase-api-2.png" subtitle="Keras는 기계가 아닌 인간을 위해 설계된 API입니다. Keras는 인지 부하를 줄이기 위한 모범 사례를 따릅니다. 일관되고 간단한 API를 제공하고, 일반적인 사용 사례에 필요한 사용자 작업 수를 최소화하며, 명확하고 실행 가능한 오류 메시지를 제공합니다. Keras는 또한 훌륭한 문서와 개발자 가이드를 작성하는 데 최우선 순위를 둡니다." >}}
{{< main_large_card title="프레임워크 선택성을 잠금 해제하세요." image="/images/framework-optionality.png" subtitle="Keras는 JAX, TensorFlow 및 PyTorch와 함께 작동합니다. 이를 통해 프레임워크 경계를 넘어 이동할 수 있고, 이러한 세 가지 프레임워크 모두의 에코시스템을 활용할 수 있는 모델을 생성할 수 있습니다." >}}
{{< main_large_card title="엑사스케일 머신러닝." image="/images/showcase-tpu.jpg" subtitle="Keras는 대규모 GPU 클러스터 또는 전체 TPU Pod로 확장할 수 있는 업계 최고의 프레임워크입니다. 그것은 가능할 뿐만 아니라, 쉽습니다." >}}
{{< main_large_card title="최첨단 연구." image="/images/showcase-lhc.jpg" subtitle="Keras는 CERN, NASA, NIH 및 전 세계의 더 많은 과학 기관에서 사용됩니다. (그리고 LHC에서도 Keras를 사용합니다) Keras는 임의의 연구 아이디어를 구현할 수 있는 낮은 수준의 유연성을 제공하는 동시에, 실험 주기를 가속화하기 위한 높은 수준의 편의 기능을 선택적으로 제공합니다." >}}
{{< /cards >}}
