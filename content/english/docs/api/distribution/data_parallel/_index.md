---
title: DataParallel API
toc: true
weight: 2
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/distribution/distribution_lib.py#L366" >}}

### `DataParallel` class

```python
keras.distribution.DataParallel(
    device_mesh=None, devices=None, auto_shard_dataset=True
)
```

Distribution for data parallelism.

You can choose to create this instance by either specifying
the `device_mesh` or `devices` arguments (but not both).

The `device_mesh` argument is expected to be a `DeviceMesh` instance,
and is expected to be 1D only. In case that the mesh has multiple axes,
then the first axis will be treated as the data parallel dimension
(and a warning will be raised).

When a list of `devices` are provided, they will be used to construct a
1D mesh.

When both `mesh` and `devices` are absent, then `list_devices()`
will be used to detect any available devices and create a 1D mesh from
them.

**Arguments**

- **device_mesh**: Optional `DeviceMesh` instance.
- **devices**: Optional list of devices.
- **auto_shard_dataset**: Automatically shard the dataset amongst processes.
  Defaults to true.
