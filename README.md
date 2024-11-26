# Keras Clone Website

This website is a clone of the Keras website.

The following changes have been made from the official Keras website:

1. Multilingual support
2. Dark mode support
3. Documentation errors fixed

## Run on local computer

This site can run locally as a Docker container.

### Docker image build

The image build is performed only once. Or, if the content is updated, it is executed.

```console
$ docker build -t keras_site_image .
```

### Docker container run

```console
$ docker run -d --name keras_site -p 8080:80 keras_site_image
```

Open `http://localhost:8080`.

## References

- Keras: https://keras.io
- Import official documentation
- Hugo: https://gohugo.io/
- Web server
- Hextra: https://imfing.github.io/hextra/
- Hugo-based documentation support plugin

## Editing on VSCode

VSCode runs on a Docker container based on `devcontainer.json`.

### VSCode API import extension

1. Press Command + Shift + P
2. Select install extension from location
3. Install the extension from this location: `/workspaces/keras_kr_doc/paste-extension/`
4. In the file, do `Insert Markdown Output`

### Regex

#### Replace relative links

```markdown
[`keras.callbacks.Callback`](/api/callbacks/base_callback#callback-class)
```

Change this link to:

```markdown
[`keras.callbacks.Callback`]({{< relref "/docs/api/callbacks/base_callback#callback-class" >}})
```

Apply the regex as follows:

- Find: `\((\/[^\)]+)\)`
- Replace: `({{< relref "/docs$1" >}})`

#### Replace keras.io links

```markdown
[`keras.callbacks.Callback`](https://keras.io/api/callbacks/base_callback#callback-class)

[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
```

Change these links to: (Do not change links other than keras.io.)

```markdown
[`keras.callbacks.Callback`]({{< relref "/docs/api/callbacks/base_callback#callback-class" >}})

[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
```

The regular expression is applied as follows.

- Find: `\[([^\]]+)\]\(https:\/\/keras\.io(\/[^\)]+)\)`
- Replace: `[$1]({{< relref "/docs$2" >}})`

#### Replace source link

```plain
[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L555)
```

Change this link to the following:

```plain
{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/backend/tensorflow/trainer.py#L555" >}}
```

The regular expression is applied as follows:

- Find: `\[\\\[source\\\]\]\((.+?)\)`
- Replace: `{{< keras/source link="$1" >}}`
