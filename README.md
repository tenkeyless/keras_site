# Keras 공식 문서

이 페이지는 [Keras 공식 문서](https://keras.io/)의 번역을 위한 것입니다.

기존 프로젝트 대신, [Hugo](https://gohugo.io/)와 [Hextra](https://imfing.github.io/hextra/)를 활용한 페이지입니다.
문서 편집을 위해 다음 사이트를 참고하세요.

- [Hextra 문서](https://imfing.github.io/hextra/docs/)

## 편집

- 상대 링크 교체
  - `\((\/[^\)]+)\)`
  - `({{< relref "/docs$1" >}})`
- keras.io 링크 교체
  - `\[([^\]]+)\]\(https:\/\/keras\.io(\/[^\)]+)\)`
  - `[$1]({{< relref "/docs$2" >}})`
- 소스 링크 교체
  - `\[\\\[source\\\]\]\((.+?)\)`
  - `{{< keras/source link="$1" >}}`

### 정규식 링크 교체

```plain
[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

[`keras.callbacks.Callback`](/api/callbacks/base_callback#callback-class)
```

이 둘에 대해 변경하고자 하는데,

```plain
[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

[`keras.callbacks.Callback`](\{\{< relref "/docs/api/callbacks/base_callback#callback-class" >\}\})
```

앞에 것은 건드리지 않고, 뒤에 것만 건드려야 하는 상황이며, 이런 경우,

```plain
\((\/[^\)]+)\)
```

이런 정규식을 검색해서,

```plain
({{< relref "/docs$1" >}})
```

이렇게 변경하도록 합니다.

### 정규식 링크 교체 2

```plain
[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L555)
```

이런 링크를

```plain
\{\{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L555" >\}\}
```

이렇게 변경하고자 할 때,

```plain
\[\\\[source\\\]\]\((.+?)\)
```

이런 정규식을 검색해서,

```plain
{{< keras/source link="$1" >}}
```

이렇게 변경하도록 합니다.
