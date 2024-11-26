---
title: RemoteMonitor
toc: true
weight: 8
type: docs
---

{{< keras/original checkedAt="2024-11-25" >}}

{{< keras/source link="https://github.com/keras-team/keras/tree/v3.6.0/keras/src/callbacks/remote_monitor.py#L15" >}}

### `RemoteMonitor` class

```python
keras.callbacks.RemoteMonitor(
    root="http://localhost:9000",
    path="/publish/epoch/end/",
    field="data",
    headers=None,
    send_as_json=False,
)
```

Callback used to stream events to a server.

Requires the `requests` library.
Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
HTTP POST, with a `data` argument which is a
JSON-encoded dictionary of event data.
If `send_as_json=True`, the content type of the request will be
`"application/json"`.
Otherwise the serialized JSON will be sent within a form.

**Arguments**

- **root**: String; root url of the target server.
- **path**: String; path relative to `root` to which the events will be sent.
- **field**: String; JSON field under which the data will be stored.
  The field is used only if the payload is sent within a form
  (i.e. when `send_as_json=False`).
- **headers**: Dictionary; optional custom HTTP headers.
- **send_as_json**: Boolean; whether the request should be
  sent as `"application/json"`.
