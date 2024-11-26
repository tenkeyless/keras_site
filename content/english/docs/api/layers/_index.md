---
title: Keras layers API
linkTitle: Layers API
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

Layers are the basic building blocks of neural networks in Keras.
A layer consists of a tensor-in tensor-out computation function (the layer's `call` method)
and some state, held in TensorFlow variables (the layer's _weights_).

A Layer instance is callable, much like a function:

```python
import keras
from keras import layers
layer = layers.Dense(32, activation='relu')
inputs = keras.random.uniform(shape=(10, 20))
outputs = layer(inputs)
```

Unlike a function, though, layers maintain a state, updated when the layer receives data
during training, and stored in `layer.weights`:

```console
>>> layer.weights
[<KerasVariable shape=(20, 32), dtype=float32, path=dense/kernel>,
 <KerasVariable shape=(32,), dtype=float32, path=dense/bias>]
```

## Creating custom layers

While Keras offers a wide range of built-in layers, they don't cover
ever possible use case. Creating custom layers is very common, and very easy.

See the guide
[Making new layers and models via subclassing]({{< relref "/docs/guides/making_new_layers_and_models_via_subclassing" >}})
for an extensive overview, and refer to the documentation for [the base `Layer` class](base_layer).

## Layers API overview

### [The base Layer class]({{< relref "/docs/api/layers/base_layer/" >}})

- [Layer class]({{< relref "/docs/api/layers/base_layer/#layer-class" >}})
- [weights property]({{< relref "/docs/api/layers/base_layer/#weights-property" >}})
- [trainable\_weights property]({{< relref "/docs/api/layers/base_layer/#trainable_weights-property" >}})
- [non\_trainable\_weights property]({{< relref "/docs/api/layers/base_layer/#non_trainable_weights-property" >}})
- [add\_weight method]({{< relref "/docs/api/layers/base_layer/#add_weight-method" >}})
- [trainable property]({{< relref "/docs/api/layers/base_layer/#trainable-property" >}})
- [get\_weights method]({{< relref "/docs/api/layers/base_layer/#get_weights-method" >}})
- [set\_weights method]({{< relref "/docs/api/layers/base_layer/#set_weights-method" >}})
- [get\_config method]({{< relref "/docs/api/layers/base_layer/#get_config-method" >}})
- [add\_loss method]({{< relref "/docs/api/layers/base_layer/#add_loss-method" >}})
- [losses property]({{< relref "/docs/api/layers/base_layer/#losses-property" >}})

### [Layer activations]({{< relref "/docs/api/layers/activations/" >}})

- [relu function]({{< relref "/docs/api/layers/activations/#relu-function" >}})
- [sigmoid function]({{< relref "/docs/api/layers/activations/#sigmoid-function" >}})
- [softmax function]({{< relref "/docs/api/layers/activations/#softmax-function" >}})
- [softplus function]({{< relref "/docs/api/layers/activations/#softplus-function" >}})
- [softsign function]({{< relref "/docs/api/layers/activations/#softsign-function" >}})
- [tanh function]({{< relref "/docs/api/layers/activations/#tanh-function" >}})
- [selu function]({{< relref "/docs/api/layers/activations/#selu-function" >}})
- [elu function]({{< relref "/docs/api/layers/activations/#elu-function" >}})
- [exponential function]({{< relref "/docs/api/layers/activations/#exponential-function" >}})
- [leaky\_relu function]({{< relref "/docs/api/layers/activations/#leaky_relu-function" >}})
- [relu6 function]({{< relref "/docs/api/layers/activations/#relu6-function" >}})
- [silu function]({{< relref "/docs/api/layers/activations/#silu-function" >}})
- [hard\_silu function]({{< relref "/docs/api/layers/activations/#hard_silu-function" >}})
- [gelu function]({{< relref "/docs/api/layers/activations/#gelu-function" >}})
- [hard\_sigmoid function]({{< relref "/docs/api/layers/activations/#hard_sigmoid-function" >}})
- [linear function]({{< relref "/docs/api/layers/activations/#linear-function" >}})
- [mish function]({{< relref "/docs/api/layers/activations/#mish-function" >}})
- [log\_softmax function]({{< relref "/docs/api/layers/activations/#log_softmax-function" >}})

### [Layer weight initializers]({{< relref "/docs/api/layers/initializers/" >}})

- [RandomNormal class]({{< relref "/docs/api/layers/initializers/#randomnormal-class" >}})
- [RandomUniform class]({{< relref "/docs/api/layers/initializers/#randomuniform-class" >}})
- [TruncatedNormal class]({{< relref "/docs/api/layers/initializers/#truncatednormal-class" >}})
- [Zeros class]({{< relref "/docs/api/layers/initializers/#zeros-class" >}})
- [Ones class]({{< relref "/docs/api/layers/initializers/#ones-class" >}})
- [GlorotNormal class]({{< relref "/docs/api/layers/initializers/#glorotnormal-class" >}})
- [GlorotUniform class]({{< relref "/docs/api/layers/initializers/#glorotuniform-class" >}})
- [HeNormal class]({{< relref "/docs/api/layers/initializers/#henormal-class" >}})
- [HeUniform class]({{< relref "/docs/api/layers/initializers/#heuniform-class" >}})
- [Orthogonal class]({{< relref "/docs/api/layers/initializers/#orthogonalinitializer-class" >}})
- [Constant class]({{< relref "/docs/api/layers/initializers/#constant-class" >}})
- [VarianceScaling class]({{< relref "/docs/api/layers/initializers/#variancescaling-class" >}})
- [LecunNormal class]({{< relref "/docs/api/layers/initializers/#lecunnormal-class" >}})
- [LecunUniform class]({{< relref "/docs/api/layers/initializers/#lecununiform-class" >}})
- [IdentityInitializer class]({{< relref "/docs/api/layers/initializers/#identity-class" >}})

### [Layer weight regularizers]({{< relref "/docs/api/layers/regularizers/" >}})

- [Regularizer class]({{< relref "/docs/api/layers/regularizers/#regularizer-class" >}})
- [L1 class]({{< relref "/docs/api/layers/regularizers/#l1-class" >}})
- [L2 class]({{< relref "/docs/api/layers/regularizers/#l2-class" >}})
- [L1L2 class]({{< relref "/docs/api/layers/regularizers/#l1l2-class" >}})
- [OrthogonalRegularizer class]({{< relref "/docs/api/layers/regularizers/#orthogonalregularizer-class" >}})

### [Layer weight constraints]({{< relref "/docs/api/layers/constraints/" >}})

- [Constraint class]({{< relref "/docs/api/layers/constraints/#constraint-class" >}})
- [MaxNorm class]({{< relref "/docs/api/layers/constraints/#maxnorm-class" >}})
- [MinMaxNorm class]({{< relref "/docs/api/layers/constraints/#minmaxnorm-class" >}})
- [NonNeg class]({{< relref "/docs/api/layers/constraints/#nonneg-class" >}})
- [UnitNorm class]({{< relref "/docs/api/layers/constraints/#unitnorm-class" >}})

### [Core layers]({{< relref "/docs/api/layers/core_layers/" >}})

- [Input object]({{< relref "/docs/api/layers/core_layers/input" >}})
- [InputSpec object]({{< relref "/docs/api/layers/core_layers/input_spec" >}})
- [Dense layer]({{< relref "/docs/api/layers/core_layers/dense" >}})
- [EinsumDense layer]({{< relref "/docs/api/layers/core_layers/einsum_dense" >}})
- [Activation layer]({{< relref "/docs/api/layers/core_layers/activation" >}})
- [Embedding layer]({{< relref "/docs/api/layers/core_layers/embedding" >}})
- [Masking layer]({{< relref "/docs/api/layers/core_layers/masking" >}})
- [Lambda layer]({{< relref "/docs/api/layers/core_layers/lambda" >}})
- [Identity layer]({{< relref "/docs/api/layers/core_layers/identity" >}})

### [Convolution layers]({{< relref "/docs/api/layers/convolution_layers/" >}})

- [Conv1D layer]({{< relref "/docs/api/layers/convolution_layers/convolution1d" >}})
- [Conv2D layer]({{< relref "/docs/api/layers/convolution_layers/convolution2d" >}})
- [Conv3D layer]({{< relref "/docs/api/layers/convolution_layers/convolution3d" >}})
- [SeparableConv1D layer]({{< relref "/docs/api/layers/convolution_layers/separable_convolution1d" >}})
- [SeparableConv2D layer]({{< relref "/docs/api/layers/convolution_layers/separable_convolution2d" >}})
- [DepthwiseConv1D layer]({{< relref "/docs/api/layers/convolution_layers/depthwise_convolution1d" >}})
- [DepthwiseConv2D layer]({{< relref "/docs/api/layers/convolution_layers/depthwise_convolution2d" >}})
- [Conv1DTranspose layer]({{< relref "/docs/api/layers/convolution_layers/convolution1d_transpose" >}})
- [Conv2DTranspose layer]({{< relref "/docs/api/layers/convolution_layers/convolution2d_transpose" >}})
- [Conv3DTranspose layer]({{< relref "/docs/api/layers/convolution_layers/convolution3d_transpose" >}})

### [Pooling layers]({{< relref "/docs/api/layers/pooling_layers/" >}})

- [MaxPooling1D layer]({{< relref "/docs/api/layers/pooling_layers/max_pooling1d" >}})
- [MaxPooling2D layer]({{< relref "/docs/api/layers/pooling_layers/max_pooling2d" >}})
- [MaxPooling3D layer]({{< relref "/docs/api/layers/pooling_layers/max_pooling3d" >}})
- [AveragePooling1D layer]({{< relref "/docs/api/layers/pooling_layers/average_pooling1d" >}})
- [AveragePooling2D layer]({{< relref "/docs/api/layers/pooling_layers/average_pooling2d" >}})
- [AveragePooling3D layer]({{< relref "/docs/api/layers/pooling_layers/average_pooling3d" >}})
- [GlobalMaxPooling1D layer]({{< relref "/docs/api/layers/pooling_layers/global_max_pooling1d" >}})
- [GlobalMaxPooling2D layer]({{< relref "/docs/api/layers/pooling_layers/global_max_pooling2d" >}})
- [GlobalMaxPooling3D layer]({{< relref "/docs/api/layers/pooling_layers/global_max_pooling3d" >}})
- [GlobalAveragePooling1D layer]({{< relref "/docs/api/layers/pooling_layers/global_average_pooling1d" >}})
- [GlobalAveragePooling2D layer]({{< relref "/docs/api/layers/pooling_layers/global_average_pooling2d" >}})
- [GlobalAveragePooling3D layer]({{< relref "/docs/api/layers/pooling_layers/global_average_pooling3d" >}})

### [Recurrent layers]({{< relref "/docs/api/layers/recurrent_layers/" >}})

- [LSTM layer]({{< relref "/docs/api/layers/recurrent_layers/lstm" >}})
- [LSTM cell layer]({{< relref "/docs/api/layers/recurrent_layers/lstm_cell" >}})
- [GRU layer]({{< relref "/docs/api/layers/recurrent_layers/gru" >}})
- [GRU Cell layer]({{< relref "/docs/api/layers/recurrent_layers/gru_cell" >}})
- [SimpleRNN layer]({{< relref "/docs/api/layers/recurrent_layers/simple_rnn" >}})
- [TimeDistributed layer]({{< relref "/docs/api/layers/recurrent_layers/time_distributed" >}})
- [Bidirectional layer]({{< relref "/docs/api/layers/recurrent_layers/bidirectional" >}})
- [ConvLSTM1D layer]({{< relref "/docs/api/layers/recurrent_layers/conv_lstm1d" >}})
- [ConvLSTM2D layer]({{< relref "/docs/api/layers/recurrent_layers/conv_lstm2d" >}})
- [ConvLSTM3D layer]({{< relref "/docs/api/layers/recurrent_layers/conv_lstm3d" >}})
- [Base RNN layer]({{< relref "/docs/api/layers/recurrent_layers/rnn" >}})
- [Simple RNN cell layer]({{< relref "/docs/api/layers/recurrent_layers/simple_rnn_cell" >}})
- [Stacked RNN cell layer]({{< relref "/docs/api/layers/recurrent_layers/stacked_rnn_cell" >}})

### [Preprocessing layers]({{< relref "/docs/api/layers/preprocessing_layers/" >}})

- [Text preprocessing]({{< relref "/docs/api/layers/preprocessing_layers/text/" >}})
- [Numerical features preprocessing layers]({{< relref "/docs/api/layers/preprocessing_layers/numerical/" >}})
- [Categorical features preprocessing layers]({{< relref "/docs/api/layers/preprocessing_layers/categorical/" >}})
- [Image preprocessing layers]({{< relref "/docs/api/layers/preprocessing_layers/image_preprocessing/" >}})
- [Image augmentation layers]({{< relref "/docs/api/layers/preprocessing_layers/image_augmentation/" >}})
- [Audio preprocessing layers]({{< relref "/docs/api/layers/preprocessing_layers/audio_preprocessing/" >}})

### [Normalization layers]({{< relref "/docs/api/layers/normalization_layers/" >}})

- [BatchNormalization layer]({{< relref "/docs/api/layers/normalization_layers/batch_normalization" >}})
- [LayerNormalization layer]({{< relref "/docs/api/layers/normalization_layers/layer_normalization" >}})
- [UnitNormalization layer]({{< relref "/docs/api/layers/normalization_layers/unit_normalization" >}})
- [GroupNormalization layer]({{< relref "/docs/api/layers/normalization_layers/group_normalization" >}})

### [Regularization layers]({{< relref "/docs/api/layers/regularization_layers/" >}})

- [Dropout layer]({{< relref "/docs/api/layers/regularization_layers/dropout" >}})
- [SpatialDropout1D layer]({{< relref "/docs/api/layers/regularization_layers/spatial_dropout1d" >}})
- [SpatialDropout2D layer]({{< relref "/docs/api/layers/regularization_layers/spatial_dropout2d" >}})
- [SpatialDropout3D layer]({{< relref "/docs/api/layers/regularization_layers/spatial_dropout3d" >}})
- [GaussianDropout layer]({{< relref "/docs/api/layers/regularization_layers/gaussian_dropout" >}})
- [AlphaDropout layer]({{< relref "/docs/api/layers/regularization_layers/alpha_dropout" >}})
- [GaussianNoise layer]({{< relref "/docs/api/layers/regularization_layers/gaussian_noise" >}})
- [ActivityRegularization layer]({{< relref "/docs/api/layers/regularization_layers/activity_regularization" >}})

### [Attention layers]({{< relref "/docs/api/layers/attention_layers/" >}})

- [GroupQueryAttention]({{< relref "/docs/api/layers/attention_layers/group_query_attention" >}})
- [MultiHeadAttention layer]({{< relref "/docs/api/layers/attention_layers/multi_head_attention" >}})
- [Attention layer]({{< relref "/docs/api/layers/attention_layers/attention" >}})
- [AdditiveAttention layer]({{< relref "/docs/api/layers/attention_layers/additive_attention" >}})

### [Reshaping layers]({{< relref "/docs/api/layers/reshaping_layers/" >}})

- [Reshape layer]({{< relref "/docs/api/layers/reshaping_layers/reshape" >}})
- [Flatten layer]({{< relref "/docs/api/layers/reshaping_layers/flatten" >}})
- [RepeatVector layer]({{< relref "/docs/api/layers/reshaping_layers/repeat_vector" >}})
- [Permute layer]({{< relref "/docs/api/layers/reshaping_layers/permute" >}})
- [Cropping1D layer]({{< relref "/docs/api/layers/reshaping_layers/cropping1d" >}})
- [Cropping2D layer]({{< relref "/docs/api/layers/reshaping_layers/cropping2d" >}})
- [Cropping3D layer]({{< relref "/docs/api/layers/reshaping_layers/cropping3d" >}})
- [UpSampling1D layer]({{< relref "/docs/api/layers/reshaping_layers/up_sampling1d" >}})
- [UpSampling2D layer]({{< relref "/docs/api/layers/reshaping_layers/up_sampling2d" >}})
- [UpSampling3D layer]({{< relref "/docs/api/layers/reshaping_layers/up_sampling3d" >}})
- [ZeroPadding1D layer]({{< relref "/docs/api/layers/reshaping_layers/zero_padding1d" >}})
- [ZeroPadding2D layer]({{< relref "/docs/api/layers/reshaping_layers/zero_padding2d" >}})
- [ZeroPadding3D layer]({{< relref "/docs/api/layers/reshaping_layers/zero_padding3d" >}})

### [Merging layers]({{< relref "/docs/api/layers/merging_layers/" >}})

- [Concatenate layer]({{< relref "/docs/api/layers/merging_layers/concatenate" >}})
- [Average layer]({{< relref "/docs/api/layers/merging_layers/average" >}})
- [Maximum layer]({{< relref "/docs/api/layers/merging_layers/maximum" >}})
- [Minimum layer]({{< relref "/docs/api/layers/merging_layers/minimum" >}})
- [Add layer]({{< relref "/docs/api/layers/merging_layers/add" >}})
- [Subtract layer]({{< relref "/docs/api/layers/merging_layers/subtract" >}})
- [Multiply layer]({{< relref "/docs/api/layers/merging_layers/multiply" >}})
- [Dot layer]({{< relref "/docs/api/layers/merging_layers/dot" >}})

### [Activation layers]({{< relref "/docs/api/layers/activation_layers/" >}})

- [ReLU layer]({{< relref "/docs/api/layers/activation_layers/relu" >}})
- [Softmax layer]({{< relref "/docs/api/layers/activation_layers/softmax" >}})
- [LeakyReLU layer]({{< relref "/docs/api/layers/activation_layers/leaky_relu" >}})
- [PReLU layer]({{< relref "/docs/api/layers/activation_layers/prelu" >}})
- [ELU layer]({{< relref "/docs/api/layers/activation_layers/elu" >}})

### [Backend-specific layers]({{< relref "/docs/api/layers/backend_specific_layers/" >}})

- [TorchModuleWrapper layer]({{< relref "/docs/api/layers/backend_specific_layers/torch_module_wrapper" >}})
- [Tensorflow SavedModel layer]({{< relref "/docs/api/layers/backend_specific_layers/tfsm_layer" >}})
- [JaxLayer]({{< relref "/docs/api/layers/backend_specific_layers/jax_layer" >}})
- [FlaxLayer]({{< relref "/docs/api/layers/backend_specific_layers/flax_layer" >}})
