# Keras 클론 웹사이트

이 웹사이트는 Keras 웹사이트의 복제본입니다.

공식 Keras 웹사이트로부터 변경된 점은 다음과 같습니다.

1. 다국어 지원
2. 다크모드 지원
3. 오류있는 문서 수정

## 로컬 컴퓨터에서 실행

이 사이트는 로컬에서 Docker 컨테이너로 실행될 수 있습니다.

### Docker 이미지 빌드

이미지 빌드는 최초 1회만 진행합니다. 혹은 내용이 업데이트 된 경우, 실행합니다.

```console
$ docker build -t keras_site_image .
```

### Docker 컨테이너 실행

```console
$ docker run -d --name keras_site -p 8080:80 keras_site_image
```

`http://localhost:8080`으로 접속합니다.

## 참고 사이트

- Keras: https://keras.io
  - 공식 문서 임포트
- Hugo: https://gohugo.io/
  - 웹 서버
- Hextra: https://imfing.github.io/hextra/
  - Hugo 기반의 문서 지원 플러그인

## VSCode 상에서의 편집

VSCode가 `devcontainer.json`를 기반으로 도커 컨테이너에서 수행됩니다.

### VSCode API import extension

1. Command + Shift + P를 누르고
2. install extension from location을 선택한 다음
3. `/workspaces/keras_kr_doc/paste-extension/` 이 위치에서 확장을 설치합니다.
4. 파일에서, `Insert Markdown Output`을 수행합니다.

### 정규식

#### 상대 링크 교체

```markdown
[`keras.callbacks.Callback`](/api/callbacks/base_callback#callback-class)
```

이런 링크를 다음과 같이 변경합니다.

```markdown
[`keras.callbacks.Callback`]({{< relref "/docs/api/callbacks/base_callback#callback-class" >}})
```

정규식은 다음과 같이 적용합니다.

- Find: `\((\/[^\)]+)\)`
- Replace: `({{< relref "/docs$1" >}})`

#### keras.io 링크 교체

```markdown
[`keras.callbacks.Callback`](https://keras.io/api/callbacks/base_callback#callback-class)

[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
```

이런 링크를 다음과 같이 변경합니다. (keras.io가 아닌 다른 링크는 변경하지 않습니다.)

```markdown
[`keras.callbacks.Callback`]({{< relref "/docs/api/callbacks/base_callback#callback-class" >}})

[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
```

정규식은 다음과 같이 적용합니다.

- Find: `\[([^\]]+)\]\(https:\/\/keras\.io(\/[^\)]+)\)`
- Replace: `[$1]({{< relref "/docs$2" >}})`

#### Source 링크 교체

```plain
[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L555)
```

이런 링크를 다음과 같이 변경합니다.

```plain
{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L555" >}}
```

정규식은 다음과 같이 적용합니다.

- Find: `\[\\\[source\\\]\]\((.+?)\)`
- Replace: `{{< keras/source link="$1" >}}`
