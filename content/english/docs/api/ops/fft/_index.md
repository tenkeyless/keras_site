---
title: FFT ops
toc: true
weight: 6
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L378" >}}

### `fft` function

```python
keras.ops.fft(x)
```

Computes the Fast Fourier Transform along last axis of input.

**Arguments**

- **x**: Tuple of the real and imaginary parts of the input tensor. Both
  tensors in the tuple should be of floating type.

**Returns**

A tuple containing two tensors - the real and imaginary parts of the
output tensor.

**Example**

```console
>>> x = (
...     keras.ops.convert_to_tensor([1., 2.]),
...     keras.ops.convert_to_tensor([0., 1.]),
... )
>>> fft(x)
(array([ 3., -1.], dtype=float32), array([ 1., -1.], dtype=float32))
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L450" >}}

### `fft2` function

```python
keras.ops.fft2(x)
```

Computes the 2D Fast Fourier Transform along the last two axes of input.

**Arguments**

- **x**: Tuple of the real and imaginary parts of the input tensor. Both
  tensors in the tuple should be of floating type.

**Returns**

A tuple containing two tensors - the real and imaginary parts of the
output.

**Example**

```console
>>> x = (
...     keras.ops.convert_to_tensor([[1., 2.], [2., 1.]]),
...     keras.ops.convert_to_tensor([[0., 1.], [1., 0.]]),
... )
>>> fft2(x)
(array([[ 6.,  0.],
    [ 0., -2.]], dtype=float32), array([[ 2.,  0.],
    [ 0., -2.]], dtype=float32))
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L509" >}}

### `rfft` function

```python
keras.ops.rfft(x, fft_length=None)
```

Real-valued Fast Fourier Transform along the last axis of the input.

Computes the 1D Discrete Fourier Transform of a real-valued signal over the
inner-most dimension of input.

Since the Discrete Fourier Transform of a real-valued signal is
Hermitian-symmetric, RFFT only returns the `fft_length / 2 + 1` unique
components of the FFT: the zero-frequency term, followed by the
`fft_length / 2` positive-frequency terms.

Along the axis RFFT is computed on, if `fft_length` is smaller than the
corresponding dimension of the input, the dimension is cropped. If it is
larger, the dimension is padded with zeros.

**Arguments**

- **x**: Input tensor.
- **fft_length**: An integer representing the number of the fft length. If not
  specified, it is inferred from the length of the last axis of `x`.
  Defaults to `None`.

**Returns**

A tuple containing two tensors - the real and imaginary parts of the
output.

**Examples**

```console
>>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
>>> rfft(x)
(array([10.0, -2.5, -2.5]), array([0.0, 3.4409548, 0.81229924]))
```

```console
>>> rfft(x, 3)
(array([3.0, -1.5]), array([0.0, 0.8660254]))
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L677" >}}

### `stft` function

```python
keras.ops.stft(
    x, sequence_length, sequence_stride, fft_length, window="hann", center=True
)
```

Short-Time Fourier Transform along the last axis of the input.

The STFT computes the Fourier transform of short overlapping windows of the
input. This giving frequency components of the signal as they change over
time.

**Arguments**

- **x**: Input tensor.
- **sequence_length**: An integer representing the sequence length.
- **sequence_stride**: An integer representing the sequence hop size.
- **fft_length**: An integer representing the size of the FFT to apply. If not
  specified, uses the smallest power of 2 enclosing `sequence_length`.
- **window**: A string, a tensor of the window or `None`. If `window` is a
  string, available values are `"hann"` and `"hamming"`. If `window`
  is a tensor, it will be used directly as the window and its length
  must be `sequence_length`. If `window` is `None`, no windowing is
  used. Defaults to `"hann"`.
- **center**: Whether to pad `x` on both sides so that the t-th sequence is
  centered at time `t * sequence_stride`. Otherwise, the t-th sequence
  begins at time `t * sequence_stride`. Defaults to `True`.

**Returns**

A tuple containing two tensors - the real and imaginary parts of the
STFT output.

**Example**

```console
>>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
>>> stft(x, 3, 2, 3)
(array([[0.75, -0.375],
   [3.75, -1.875],
   [5.25, -2.625]]), array([[0.0, 0.64951905],
   [0.0, 0.64951905],
   [0.0, -0.64951905]]))
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L590" >}}

### `irfft` function

```python
keras.ops.irfft(x, fft_length=None)
```

Inverse real-valued Fast Fourier transform along the last axis.

Computes the inverse 1D Discrete Fourier Transform of a real-valued signal
over the inner-most dimension of input.

The inner-most dimension of the input is assumed to be the result of RFFT:
the `fft_length / 2 + 1` unique components of the DFT of a real-valued
signal. If `fft_length` is not provided, it is computed from the size of the
inner-most dimension of the input `(fft_length = 2 * (inner - 1))`. If the
FFT length used to compute is odd, it should be provided since it cannot
be inferred properly.

Along the axis IRFFT is computed on, if `fft_length / 2 + 1` is smaller than
the corresponding dimension of the input, the dimension is cropped. If it is
larger, the dimension is padded with zeros.

**Arguments**

- **x**: Tuple of the real and imaginary parts of the input tensor. Both
  tensors in the tuple should be of floating type.
- **fft_length**: An integer representing the number of the fft length. If not
  specified, it is inferred from the length of the last axis of `x`.
  Defaults to `None`.

**Returns**

A tensor containing the inverse real-valued Fast Fourier Transform
along the last axis of `x`.

**Examples**

```console
>>> real = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
>>> imag = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
>>> irfft((real, imag))
array([0.66666667, -0.9106836, 0.24401694])
```

```console
>>> irfft(rfft(real, 5), 5)
array([0.0, 1.0, 2.0, 3.0, 4.0])
```

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/ops/math.py#L797" >}}

### `istft` function

```python
keras.ops.istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
)
```

Inverse Short-Time Fourier Transform along the last axis of the input.

To reconstruct an original waveform, the parameters should be the same in
`stft`.

**Arguments**

- **x**: Tuple of the real and imaginary parts of the input tensor. Both
  tensors in the tuple should be of floating type.
- **sequence_length**: An integer representing the sequence length.
- **sequence_stride**: An integer representing the sequence hop size.
- **fft_length**: An integer representing the size of the FFT that produced
  `stft`.
- **length**: An integer representing the output is clipped to exactly length.
  If not specified, no padding or clipping take place. Defaults to
  `None`.
- **window**: A string, a tensor of the window or `None`. If `window` is a
  string, available values are `"hann"` and `"hamming"`. If `window`
  is a tensor, it will be used directly as the window and its length
  must be `sequence_length`. If `window` is `None`, no windowing is
  used. Defaults to `"hann"`.
- **center**: Whether `x` was padded on both sides so that the t-th sequence
  is centered at time `t * sequence_stride`. Defaults to `True`.

**Returns**

A tensor containing the inverse Short-Time Fourier Transform along the
last axis of `x`.

**Example**

```console
>>> x = keras.ops.convert_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
>>> istft(stft(x, 1, 1, 1), 1, 1, 1)
array([0.0, 1.0, 2.0, 3.0, 4.0])
```
