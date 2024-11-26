---
title: HyperImageAugment
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras-tuner/tree/v1.4.7/keras_tuner/src/applications/augment.py#L69" >}}

### `HyperImageAugment` class

```python
keras_tuner.applications.HyperImageAugment(
    input_shape=None,
    input_tensor=None,
    rotate=0.5,
    translate_x=0.4,
    translate_y=0.4,
    contrast=0.3,
    augment_layers=3,
    **kwargs
)
```

A image augmentation hypermodel.

The `HyperImageAugment` class searches for the best combination of image
augmentation operations in Keras preprocessing layers. The input shape of
the model should be (height, width, channels). The output of the model is
of the same shape as the input.

**Arguments**

- **input_shape**: Optional shape tuple, e.g. `(256, 256, 3)`.
- **input_tensor**: Optional Keras tensor (i.e. output of `layers.Input()`)
  to use as image input for the model.
- **rotate**: A number between [0, 1], a list of two numbers between [0, 1]
  or None. Configures the search space of the factor of random
  rotation transform in the augmentation. A factor is chosen for each
  trial. It sets maximum of clockwise and counterclockwise rotation
  in terms of fraction of pi, among all samples in the trial.
  Default is 0.5. When `rotate` is a single number, the search range
  is [0, `rotate`].
  The transform is off when set to None.
- **translate_x**: A number between [0, 1], a list of two numbers between
  [0, 1] or None. Configures the search space of the factor of random
  horizontal translation transform in the augmentation. A factor is
  chosen for each trial. It sets maximum of horizontal translation in
  terms of ratio over the width among all samples in the trial.
  Default is 0.4. When `translate_x` is a single number, the search
  range is [0, `translate_x`].
  The transform is off when set to None.
- **translate_y**: A number between [0, 1], a list of two numbers between
  [0, 1] or None. Configures the search space of the factor of random
  vertical translation transform in the augmentation. A factor is
  chosen for each trial. It sets maximum of vertical translation in
  terms of ratio over the height among all samples in the trial.
  Default is 0.4. When `translate_y` is a single number ,the search
  range is [0, `translate_y`]. The transform is off when set to None.
- **contrast**: A number between [0, 1], a list of two numbers between [0, 1]
  or None. Configures the search space of the factor of random
  contrast transform in the augmentation. A factor is chosen for each
  trial. It sets maximum ratio of contrast change among all samples in
  the trial. Default is 0.3. When `contrast` is a single number, the
  search rnage is [0, `contrast`].
  The transform is off when set to None.
- **augment_layers**: None, int or list of two ints, controlling the number
  of augment applied. Default is 3.
  When `augment_layers` is 0, all transform are applied sequentially.
  When `augment_layers` is nonzero, or a list of two ints, a simple
  version of RandAugment(https://arxiv.org/abs/1909.13719) is used.
  A search space for 'augment_layers' is created to search [0,
  `augment_layers`], or between the two ints if a `augment_layers` is
  a list. For each trial, the hyperparameter 'augment_layers'
  determines number of layers of augment transforms are applied,
  each randomly picked from all available transform types with equal
  probability on each sample.
- **\*\*kwargs**: Additional keyword arguments that apply to all hypermodels.
  See [`keras_tuner.HyperModel`]({{< relref "/docs/api/keras_tuner/hypermodels/base_hypermodel#hypermodel-class" >}}).

**Example**

```python
hm_aug = HyperImageAugment(input_shape=(32, 32, 3),
                           augment_layers=0,
                           rotate=[0.2, 0.3],
                           translate_x=0.1,
                           translate_y=None,
                           contrast=None)
```

Then the hypermodel `hm_aug` will search 'factor_rotate' between [0.2, 0.3]
and 'factor_translate_x' between [0, 0.1]. These two augments are applied
on all samples with factor picked per each trial.

```python
hm_aug = HyperImageAugment(input_shape=(32, 32, 3),
                           translate_x=0.5,
                           translate_y=[0.2, 0.4]
                           contrast=None)
```

Then the hypermodel `hm_aug` will search 'factor_rotate' between [0, 0.2],
'factor_translate_x' between [0, 0.5], 'factor_translate_y' between
[0.2, 0.4]. It will use RandAugment, searching 'augment_layers'
between [0, 3]. Each layer on each sample will be chosen from rotate,
translate_x and translate_y.
