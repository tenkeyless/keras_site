---
title: stable_diffusion
toc: false
---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/stable_diffusion/stable_diffusion.py#L361)

### `StableDiffusion` class

`keras_cv.models.StableDiffusion(img_height=512, img_width=512, jit_compile=True)`

Keras implementation of Stable Diffusion.

Note that the StableDiffusion API, as well as the APIs of the sub-components of StableDiffusion (e.g. ImageEncoder, DiffusionModel) should be considered unstable at this point. We do not guarantee backwards compatability for future changes to these APIs.

Stable Diffusion is a powerful image generation model that can be used, among other things, to generate pictures according to a short text description (called a "prompt").

**Arguments**

- **img_height**: int, height of the images to generate, in pixel. Note that only multiples of 128 are supported; the value provided will be rounded to the nearest valid value. Defaults to 512.
- **img_width**: int, width of the images to generate, in pixel. Note that only multiples of 128 are supported; the value provided will be rounded to the nearest valid value. Defaults to 512.
- **jit_compile**: bool, whether to compile the underlying models to XLA. This can lead to a significant speedup on some systems. Defaults to False.

**Example**

`from keras_cv.src.models import StableDiffusion from PIL import Image  model = StableDiffusion(img_height=512, img_width=512, jit_compile=True) img = model.text_to_image(     prompt="A beautiful horse running through a field",     batch_size=1,  # How many images to generate at once     num_steps=25,  # Number of iterations (controls image quality)     seed=123,  # Set this to always get the same image from the same prompt ) Image.fromarray(img[0]).save("horse.png") print("saved at horse.png")`

**References**

- [About Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement)
- [Original implementation](https://github.com/CompVis/stable-diffusion)

---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/stable_diffusion/stable_diffusion.py#L447)

### `StableDiffusionV2` class

`keras_cv.models.StableDiffusionV2(img_height=512, img_width=512, jit_compile=True)`

Keras implementation of Stable Diffusion v2.

Note that the StableDiffusion API, as well as the APIs of the sub-components of StableDiffusionV2 (e.g. ImageEncoder, DiffusionModelV2) should be considered unstable at this point. We do not guarantee backwards compatability for future changes to these APIs.

Stable Diffusion is a powerful image generation model that can be used, among other things, to generate pictures according to a short text description (called a "prompt").

**Arguments**

- **img_height**: int, height of the images to generate, in pixel. Note that only multiples of 128 are supported; the value provided will be rounded to the nearest valid value. Defaults to 512.
- **img_width**: int, width of the images to generate, in pixel. Note that only multiples of 128 are supported; the value provided will be rounded to the nearest valid value. Defaults to 512.
- **jit_compile**: bool, whether to compile the underlying models to XLA. This can lead to a significant speedup on some systems. Defaults to False.

**Example**

`from keras_cv.src.models import StableDiffusionV2 from PIL import Image  model = StableDiffusionV2(img_height=512, img_width=512, jit_compile=True) img = model.text_to_image(     prompt="A beautiful horse running through a field",     batch_size=1,  # How many images to generate at once     num_steps=25,  # Number of iterations (controls image quality)     seed=123,  # Set this to always get the same image from the same prompt ) Image.fromarray(img[0]).save("horse.png") print("saved at horse.png")`

**References**

- [About Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement)
- [Original implementation](https://github.com/Stability-AI/stablediffusion)

---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/stable_diffusion/decoder.py#L24)

### `Decoder` class

`keras_cv.models.stable_diffusion.Decoder(     img_height, img_width, name=None, download_weights=True )`

`Sequential` groups a linear stack of layers into a `Model`.

**Examples**

`` model = keras.Sequential() model.add(keras.Input(shape=(16,))) model.add(keras.layers.Dense(8))  # Note that you can also omit the initial `Input`. # In that case the model doesn't have any weights until the first call # to a training/evaluation method (since it isn't yet built): model = keras.Sequential() model.add(keras.layers.Dense(8)) model.add(keras.layers.Dense(4)) # model.weights not created yet  # Whereas if you specify an `Input`, the model gets built # continuously as you are adding layers: model = keras.Sequential() model.add(keras.Input(shape=(16,))) model.add(keras.layers.Dense(8)) len(model.weights)  # Returns "2"  # When using the delayed-build pattern (no input shape specified), you can # choose to manually build your model by calling # `build(batch_input_shape)`: model = keras.Sequential() model.add(keras.layers.Dense(8)) model.add(keras.layers.Dense(4)) model.build((None, 16)) len(model.weights)  # Returns "4"  # Note that when using the delayed-build pattern (no input shape specified), # the model gets built the first time you call `fit`, `eval`, or `predict`, # or the first time you call the model on some input data. model = keras.Sequential() model.add(keras.layers.Dense(8)) model.add(keras.layers.Dense(1)) model.compile(optimizer='sgd', loss='mse') # This builds the model for the first time: model.fit(x, y, batch_size=32, epochs=10) ``

---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/stable_diffusion/diffusion_model.py#L21)

### `DiffusionModel` class

`keras_cv.models.stable_diffusion.DiffusionModel(     img_height, img_width, max_text_length, name=None, download_weights=True )`

A model grouping layers into an object with training/inference features.

There are three ways to instantiate a `Model`:

## With the "Functional API"

You start from `Input`, you chain layer calls to specify the model's forward pass, and finally, you create your model from inputs and outputs:

`inputs = keras.Input(shape=(37,)) x = keras.layers.Dense(32, activation="relu")(inputs) outputs = keras.layers.Dense(5, activation="softmax")(x) model = keras.Model(inputs=inputs, outputs=outputs)`

Note: Only dicts, lists, and tuples of input tensors are supported. Nested inputs are not supported (e.g. lists of list or dicts of dict).

A new Functional API model can also be created by using the intermediate tensors. This enables you to quickly extract sub-components of the model.

**Example**

`inputs = keras.Input(shape=(None, None, 3)) processed = keras.layers.RandomCrop(width=128, height=128)(inputs) conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed) pooling = keras.layers.GlobalAveragePooling2D()(conv) feature = keras.layers.Dense(10)(pooling)  full_model = keras.Model(inputs, feature) backbone = keras.Model(processed, conv) activations = keras.Model(conv, feature)`

Note that the `backbone` and `activations` models are not created with [`keras.Input`](/api/layers/core_layers/input#input-function) objects, but with the tensors that originate from [`keras.Input`](/api/layers/core_layers/input#input-function) objects. Under the hood, the layers and weights will be shared across these models, so that user can train the `full_model`, and use `backbone` or `activations` to do feature extraction. The inputs and outputs of the model can be nested structures of tensors as well, and the created models are standard Functional API models that support all the existing APIs.

## By subclassing the `Model` class

In that case, you should define your layers in `__init__()` and you should implement the model's forward pass in `call()`.

`class MyModel(keras.Model):     def __init__(self):         super().__init__()         self.dense1 = keras.layers.Dense(32, activation="relu")         self.dense2 = keras.layers.Dense(5, activation="softmax")      def call(self, inputs):         x = self.dense1(inputs)         return self.dense2(x)  model = MyModel()`

If you subclass `Model`, you can optionally have a `training` argument (boolean) in `call()`, which you can use to specify a different behavior in training and inference:

`class MyModel(keras.Model):     def __init__(self):         super().__init__()         self.dense1 = keras.layers.Dense(32, activation="relu")         self.dense2 = keras.layers.Dense(5, activation="softmax")         self.dropout = keras.layers.Dropout(0.5)      def call(self, inputs, training=False):         x = self.dense1(inputs)         x = self.dropout(x, training=training)         return self.dense2(x)  model = MyModel()`

Once the model is created, you can config the model with losses and metrics with `model.compile()`, train the model with `model.fit()`, or use the model to do prediction with `model.predict()`.

## With the `Sequential` class

In addition, [`keras.Sequential`](/api/models/sequential#sequential-class) is a special case of model where the model is purely a stack of single-input, single-output layers.

`model = keras.Sequential([     keras.Input(shape=(None, None, 3)),     keras.layers.Conv2D(filters=32, kernel_size=3), ])`

---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/stable_diffusion/image_encoder.py#L24)

### `ImageEncoder` class

`keras_cv.models.stable_diffusion.ImageEncoder(download_weights=True)`

ImageEncoder is the VAE Encoder for StableDiffusion.

---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/stable_diffusion/noise_scheduler.py#L24)

### `NoiseScheduler` class

`keras_cv.models.stable_diffusion.NoiseScheduler(     train_timesteps=1000,     beta_start=0.0001,     beta_end=0.02,     beta_schedule="linear",     variance_type="fixed_small",     clip_sample=True, )`

# Arguments

`` train_timesteps: number of diffusion steps used to train the model. beta_start: the starting `beta` value of inference. beta_end: the final `beta` value. beta_schedule: the beta schedule, a mapping from a beta range to a     sequence of betas for stepping the model. Choose from `linear` or     `quadratic`. variance_type: options to clip the variance used when adding noise to     the de-noised sample. Choose from `fixed_small`, `fixed_small_log`,     `fixed_large`, `fixed_large_log`, `learned` or `learned_range`. clip_sample: option to clip predicted sample between -1 and 1 for     numerical stability. ``

---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/stable_diffusion/clip_tokenizer.py#L80)

### `SimpleTokenizer` class

`keras_cv.models.stable_diffusion.SimpleTokenizer(bpe_path=None)`

---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/stable_diffusion/text_encoder.py#L20)

### `TextEncoder` class

`keras_cv.models.stable_diffusion.TextEncoder(     max_length, vocab_size=49408, name=None, download_weights=True )`

A model grouping layers into an object with training/inference features.

There are three ways to instantiate a `Model`:

## With the "Functional API"

You start from `Input`, you chain layer calls to specify the model's forward pass, and finally, you create your model from inputs and outputs:

`inputs = keras.Input(shape=(37,)) x = keras.layers.Dense(32, activation="relu")(inputs) outputs = keras.layers.Dense(5, activation="softmax")(x) model = keras.Model(inputs=inputs, outputs=outputs)`

Note: Only dicts, lists, and tuples of input tensors are supported. Nested inputs are not supported (e.g. lists of list or dicts of dict).

A new Functional API model can also be created by using the intermediate tensors. This enables you to quickly extract sub-components of the model.

**Example**

`inputs = keras.Input(shape=(None, None, 3)) processed = keras.layers.RandomCrop(width=128, height=128)(inputs) conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed) pooling = keras.layers.GlobalAveragePooling2D()(conv) feature = keras.layers.Dense(10)(pooling)  full_model = keras.Model(inputs, feature) backbone = keras.Model(processed, conv) activations = keras.Model(conv, feature)`

Note that the `backbone` and `activations` models are not created with [`keras.Input`](/api/layers/core_layers/input#input-function) objects, but with the tensors that originate from [`keras.Input`](/api/layers/core_layers/input#input-function) objects. Under the hood, the layers and weights will be shared across these models, so that user can train the `full_model`, and use `backbone` or `activations` to do feature extraction. The inputs and outputs of the model can be nested structures of tensors as well, and the created models are standard Functional API models that support all the existing APIs.

## By subclassing the `Model` class

In that case, you should define your layers in `__init__()` and you should implement the model's forward pass in `call()`.

`class MyModel(keras.Model):     def __init__(self):         super().__init__()         self.dense1 = keras.layers.Dense(32, activation="relu")         self.dense2 = keras.layers.Dense(5, activation="softmax")      def call(self, inputs):         x = self.dense1(inputs)         return self.dense2(x)  model = MyModel()`

If you subclass `Model`, you can optionally have a `training` argument (boolean) in `call()`, which you can use to specify a different behavior in training and inference:

`class MyModel(keras.Model):     def __init__(self):         super().__init__()         self.dense1 = keras.layers.Dense(32, activation="relu")         self.dense2 = keras.layers.Dense(5, activation="softmax")         self.dropout = keras.layers.Dropout(0.5)      def call(self, inputs, training=False):         x = self.dense1(inputs)         x = self.dropout(x, training=training)         return self.dense2(x)  model = MyModel()`

Once the model is created, you can config the model with losses and metrics with `model.compile()`, train the model with `model.fit()`, or use the model to do prediction with `model.predict()`.

## With the `Sequential` class

In addition, [`keras.Sequential`](/api/models/sequential#sequential-class) is a special case of model where the model is purely a stack of single-input, single-output layers.

`model = keras.Sequential([     keras.Input(shape=(None, None, 3)),     keras.layers.Conv2D(filters=32, kernel_size=3), ])`

---

[\[source\]](https://github.com/keras-team/keras-cv/tree/v0.9.0/keras_cv/src/models/stable_diffusion/text_encoder.py#L45)

### `TextEncoderV2` class

`keras_cv.models.stable_diffusion.TextEncoderV2(     max_length, vocab_size=49408, name=None, download_weights=True )`

A model grouping layers into an object with training/inference features.

There are three ways to instantiate a `Model`:

## With the "Functional API"

You start from `Input`, you chain layer calls to specify the model's forward pass, and finally, you create your model from inputs and outputs:

`inputs = keras.Input(shape=(37,)) x = keras.layers.Dense(32, activation="relu")(inputs) outputs = keras.layers.Dense(5, activation="softmax")(x) model = keras.Model(inputs=inputs, outputs=outputs)`

Note: Only dicts, lists, and tuples of input tensors are supported. Nested inputs are not supported (e.g. lists of list or dicts of dict).

A new Functional API model can also be created by using the intermediate tensors. This enables you to quickly extract sub-components of the model.

**Example**

`inputs = keras.Input(shape=(None, None, 3)) processed = keras.layers.RandomCrop(width=128, height=128)(inputs) conv = keras.layers.Conv2D(filters=32, kernel_size=3)(processed) pooling = keras.layers.GlobalAveragePooling2D()(conv) feature = keras.layers.Dense(10)(pooling)  full_model = keras.Model(inputs, feature) backbone = keras.Model(processed, conv) activations = keras.Model(conv, feature)`

Note that the `backbone` and `activations` models are not created with [`keras.Input`](/api/layers/core_layers/input#input-function) objects, but with the tensors that originate from [`keras.Input`](/api/layers/core_layers/input#input-function) objects. Under the hood, the layers and weights will be shared across these models, so that user can train the `full_model`, and use `backbone` or `activations` to do feature extraction. The inputs and outputs of the model can be nested structures of tensors as well, and the created models are standard Functional API models that support all the existing APIs.

## By subclassing the `Model` class

In that case, you should define your layers in `__init__()` and you should implement the model's forward pass in `call()`.

`class MyModel(keras.Model):     def __init__(self):         super().__init__()         self.dense1 = keras.layers.Dense(32, activation="relu")         self.dense2 = keras.layers.Dense(5, activation="softmax")      def call(self, inputs):         x = self.dense1(inputs)         return self.dense2(x)  model = MyModel()`

If you subclass `Model`, you can optionally have a `training` argument (boolean) in `call()`, which you can use to specify a different behavior in training and inference:

`class MyModel(keras.Model):     def __init__(self):         super().__init__()         self.dense1 = keras.layers.Dense(32, activation="relu")         self.dense2 = keras.layers.Dense(5, activation="softmax")         self.dropout = keras.layers.Dropout(0.5)      def call(self, inputs, training=False):         x = self.dense1(inputs)         x = self.dropout(x, training=training)         return self.dense2(x)  model = MyModel()`

Once the model is created, you can config the model with losses and metrics with `model.compile()`, train the model with `model.fit()`, or use the model to do prediction with `model.predict()`.

## With the `Sequential` class

In addition, [`keras.Sequential`](/api/models/sequential#sequential-class) is a special case of model where the model is purely a stack of single-input, single-output layers.

`model = keras.Sequential([     keras.Input(shape=(None, None, 3)),     keras.layers.Conv2D(filters=32, kernel_size=3), ])`

---
