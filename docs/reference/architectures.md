# 🏗️ Architectures API

The `node_fdm.architectures` namespace contains the generic contracts and
entry-point discovery mechanism for flight-dynamics architectures.

Concrete validated specifications are distributed by `node-fdm-models`.
Research and third-party packages use the same `node_fdm.architectures` entry
point and do not modify the core library.

---

## 🧩 Registry

The registry provides direct registration for backward compatibility plus
`discover_architectures()`, `available()`, `get_origin()`, and deterministic
spec digests for installed providers.

::: node_fdm.architectures.registry
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

---

See [Create an Architecture](../howto/create_architecture/) for the provider
package contract and the paper tutorial for an executable external example.
