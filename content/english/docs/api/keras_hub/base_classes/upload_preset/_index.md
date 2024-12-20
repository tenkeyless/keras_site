---
title: upload_preset
toc: true
weight: 18
type: docs
---

{{< keras/original checkedAt="2024-11-26" >}}

{{< keras/source link="https://github.com/keras-team/keras-hub/tree/v0.17.0/keras_hub/src/utils/preset_utils.py#L365" >}}

### `upload_preset` function

```python
keras_hub.upload_preset(uri, preset)
```

Upload a preset directory to a model hub.

**Arguments**

- **uri**: The URI identifying model to upload to.
  URIs with format
  `kaggle://<KAGGLE_USERNAME>/<MODEL>/<FRAMEWORK>/<VARIATION>`
  will be uploaded to Kaggle Hub while URIs with format
  `hf://[<HF_USERNAME>/]<MODEL>` will be uploaded to the Hugging
  Face Hub.
- **preset**: The path to the local model preset directory.
