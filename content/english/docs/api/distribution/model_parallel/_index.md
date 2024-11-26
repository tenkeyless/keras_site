---
title: ModelParallel API
toc: true
weight: 3
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/distribution/distribution_lib.py#L491" >}}

### `ModelParallel` class

```python
keras.distribution.ModelParallel(layout_map=None, batch_dim_name=None, **kwargs)
```

Distribution that shards model variables.

Compare to `DataParallel` which replicates the variables across all devices,
`ModelParallel` allows you to shard variables in addition to the input data.

To construct a `ModelParallel` distribution, you need to provide a
`DeviceMesh` and a `LayoutMap`.

1. `DeviceMesh` contains physical device information. The axis names in
   the mesh will be used to map the variable and data layout.
2. `LayoutMap` contains the mapping between variable paths to their
   corresponding `TensorLayout`.

**Example**

```python
devices = list_devices()    # Assume there are 8 devices.
device_mesh = DeviceMesh(shape=(2, 4), axis_names=('batch', 'model'),
                         devices=devices)
layout_map = LayoutMap(device_mesh)
layout_map['dense.*kernel'] = (None, 'model')
layout_map['dense.*bias'] = ('model',)
layout_map['conv2d.*kernel'] = (None, None, None, 'model')
layout_map['conv2d.*bias'] = ('model',)
distribution = ModelParallel(
    layout_map=layout_map,
    batch_dim_name='batch',
)
set_distribution(distribution)
model = model_creation()
model.compile()
model.fit(data)
```

You can quickly update the device mesh shape to change the sharding factor
of the variables. E.g.

```python
device_mesh = DeviceMesh(
    shape=(1, 8),
    axis_names=('batch', 'model'),
    devices=devices,
)
```

To figure out a proper layout mapping rule for all the model variables, you
can first list out all the model variable paths, which will be used as the
key to map the variables to `TensorLayout`.

e.g.

```python
model = create_model()
for v in model.variables:
    print(v.path)
```

**Arguments**

- **layout_map**: `LayoutMap` instance which map the variable path to the
  corresponding tensor layout.
- **batch_dim_name**: Optional string, the axis name in the device mesh
  (of the `layout_map` object)
  that will be used to distribute data. If unspecified, the
  first axis from the device mesh will be used.
