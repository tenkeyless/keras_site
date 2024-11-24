---
title: LayoutMap API
toc: true
weight: 1
type: docs
---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/distribution/distribution_lib.py#L683)

### `LayoutMap` class

`keras.distribution.LayoutMap(device_mesh)`

A dict-like object that maps string to `TensorLayout` instances.

`LayoutMap` uses a string as key and a `TensorLayout` as value. There is a behavior difference between a normal Python dict and this class. The string key will be treated as a regex when retrieving the value. See the docstring of `get` for more details.

See below for a usage example. You can define the naming schema of the `TensorLayout`, and then retrieve the corresponding `TensorLayout` instance.

In the normal case, the key to query is usually the `variable.path`, which is the identifier of the variable.

As shortcut, tuple or list of axis names are also allowed when inserting as value, and will be converted to `TensorLayout`.

`layout_map = LayoutMap(device_mesh) layout_map['dense.*kernel'] = (None, 'model') layout_map['dense.*bias'] = ('model',) layout_map['conv2d.*kernel'] = (None, None, None, 'model') layout_map['conv2d.*bias'] = ('model',)  layout_1 = layout_map['dense_1.kernel']             # layout_1 == layout_2d layout_2 = layout_map['dense_1.bias']               # layout_2 == layout_1d layout_3 = layout_map['dense_2.kernel']             # layout_3 == layout_2d layout_4 = layout_map['dense_2.bias']               # layout_4 == layout_1d layout_5 = layout_map['my_model/conv2d_123/kernel'] # layout_5 == layout_4d layout_6 = layout_map['my_model/conv2d_123/bias']   # layout_6 == layout_1d layout_7 = layout_map['my_model/conv3d_1/kernel']   # layout_7 == None layout_8 = layout_map['my_model/conv3d_1/bias']     # layout_8 == None`

**Arguments**

- **device_mesh**: [`keras.distribution.DeviceMesh`](/api/distribution/layout_map#devicemesh-class) instance.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/distribution/distribution_lib.py#L134)

### `DeviceMesh` class

`keras.distribution.DeviceMesh(shape, axis_names, devices=None)`

A cluster of computation devices for distributed computation.

This API is aligned with `jax.sharding.Mesh` and [`tf.dtensor.Mesh`](https://www.tensorflow.org/api_docs/python/tf/dtensor/Mesh), which represents the computation devices in the global context.

See more details in [jax.sharding.Mesh](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh) and [tf.dtensor.Mesh](https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Mesh).

**Arguments**

- **shape**: tuple of list of integers. The shape of the overall `DeviceMesh`, e.g. `(8,)` for a data parallel only distribution, or `(4, 2)` for a model+data parallel distribution.
- **axis_names**: List of string. The logical name of the each axis for the `DeviceMesh`. The length of the `axis_names` should match to the rank of the `shape`. The `axis_names` will be used to match/create the `TensorLayout` when distribute the data and variables.
- **devices**: Optional list of devices. Defaults to all the available devices locally from `keras.distribution.list_devices()`.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/distribution/distribution_lib.py#L212)

### `TensorLayout` class

`keras.distribution.TensorLayout(axes, device_mesh=None)`

A layout to apply to a tensor.

This API is aligned with `jax.sharding.NamedSharding` and [`tf.dtensor.Layout`](https://www.tensorflow.org/api_docs/python/tf/dtensor/Layout).

See more details in [jax.sharding.NamedSharding](https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.NamedSharding) and [tf.dtensor.Layout](https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Layout).

**Arguments**

- **axes**: tuple of strings that should map to the `axis_names` in a `DeviceMesh`. For any dimensions that doesn't need any sharding, A `None` can be used a placeholder.
- **device_mesh**: Optional `DeviceMesh` that will be used to create the layout. The actual mapping of tensor to physical device is not known until the mesh is specified.

---

[\[source\]](https://github.com/keras-team/keras/tree/v3.6.0/keras/src/distribution/distribution_lib.py#L805)

### `distribute_tensor` function

`keras.distribution.distribute_tensor(tensor, layout)`

Change the layout of a Tensor value in the jit function execution.

**Arguments**

- **tensor**: a Tensor to change the layout.
- **layout**: `TensorLayout` to be applied on the value.

**Returns**

a new value with the specified tensor layout.

---
