---
title: grid_mask
toc: false
---

{{< keras/source link="https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/layers/preprocessing/grid_mask.py#L37" >}}

### `GridMask` class

`keras_cv.layers.GridMask(     ratio_factor=(0, 0.5),     rotation_factor=0.15,     fill_mode="constant",     fill_value=0.0,     seed=None,     **kwargs )`

GridMask class for grid-mask augmentation.

**Input shape**

Int or float tensor with values in the range \[0, 255\]. 3D (unbatched) or 4D (batched) tensor with shape: `(..., height, width, channels)`, in `"channels_last"` format

**Output shape**

3D (unbatched) or 4D (batched) tensor with shape: `(..., height, width, channels)`, in `"channels_last"` format

**Arguments**

- **ratio_factor**: A float, tuple of two floats, or `keras_cv.FactorSampler`. Ratio determines the ratio from spacings to grid masks. Lower values make the grid size smaller, and higher values make the grid mask large. Floats should be in the range \[0, 1\]. 0.5 indicates that grid and spacing will be of equal size. To always use the same value, pass a `keras_cv.src.ConstantFactorSampler()`.

  Defaults to `(0, 0.5)`. - **rotation_factor**: The rotation_factor will be used to randomly rotate the grid_mask during training. Default to 0.1, which results in an output rotating by a random amount in the range \[-10% \* 2pi, 10% \* 2pi\].

  A float represented as fraction of 2 Pi, or a tuple of size 2 representing lower and upper bound for rotating clockwise and counter-clockwise. A positive values means rotating counter clock-wise, while a negative value means clock-wise. When represented as a single float, this value is used for both the upper and lower bound. For instance, factor=(-0.2, 0.3) results in an output rotation by a random amount in the range \[-20% \* 2pi, 30% \* 2pi\]. factor=0.2 results in an output rotating by a random amount in the range \[-20% \* 2pi, 20% \* 2pi\]. - \_\_ fill_mode\_\_: Pixels inside the gridblock are filled according to the given mode (one of `{"constant", "gaussian_noise"}`), defaults to "constant". - _constant_: Pixels are filled with the same constant value. - _gaussian_noise_: Pixels are filled with random gaussian noise. - **fill_value**: an integer represents of value to be filled inside the gridblock when `fill_mode="constant"`. Valid integer range \[0 to 255\] - **seed**: Integer. Used to create a random seed.

**Example**

`(images, labels), _ = keras.datasets.cifar10.load_data() random_gridmask = keras_cv.layers.preprocessing.GridMask() augmented_images = random_gridmask(images)`

**References**

- [GridMask paper](https://arxiv.org/abs/2001.04086)

---
