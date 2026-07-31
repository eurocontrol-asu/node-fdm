# 🏗️ Create an Architecture Outside the Library

Architectures, their schemas, preprocessing hooks, and model-specific layers
belong in a separate Python package. You should not fork or edit `node-fdm` to
add one.

The complete executable example lives in
`papers/paper_node_fdm_v2/`. The steps below describe the same contract.

## 1. Create a provider package

Use a normal `src/` layout:

```text
my-fdm-models/
├── pyproject.toml
└── src/my_fdm_models/
    ├── architecture.py
    ├── catalog.py
    ├── layers.py
    ├── preprocessing.py
    └── schema.py
```

Keep column constants in `schema.py` and preprocessing in your own package.
Only reusable, dataset-independent transformations belong in `node-fdm-data`.

## 2. Define the spec and optional custom layers

```python title="src/my_fdm_models/architecture.py"
from node_fdm.architectures import ArchitectureSpec, LayerSpec

MY_ARCH = ArchitectureSpec(
    name="my_arch_v1",
    x_cols=["altitude_m", "vertical_speed_ms"],
    u_cols=["acceleration_command_ms2"],
    e0_cols=[],
    e1_cols=[],
    dx_cols=[(1, "d_altitude_ms"), (1, "d_vertical_speed_ms2")],
    layers=[
        LayerSpec(
            name="kinematics",
            layer_class="my_fdm_models.layers.KinematicLayer",
            input_cols=[
                "altitude_m",
                "vertical_speed_ms",
                "acceleration_command_ms2",
            ],
            output_cols=["d_altitude_ms", "d_vertical_speed_ms2"],
            trainable=False,
        ),
    ],
    preprocessing_fn="my_fdm_models.preprocessing.preprocess",
)
```

`layer_class`, `preprocessing_fn`, and `segment_filter_fn` are dotted paths, so
their implementations may live entirely outside the core libraries.

## 3. Expose a catalog entry point

```python title="src/my_fdm_models/catalog.py"
from node_fdm.architectures import ArchitectureSpec
from my_fdm_models.architecture import MY_ARCH


def architectures() -> dict[str, ArchitectureSpec]:
    return {
        "my_arch": MY_ARCH,
        MY_ARCH.name: MY_ARCH,
    }
```

Declare the provider in `pyproject.toml`:

```toml
[project.entry-points."node_fdm.architectures"]
my_models = "my_fdm_models.catalog:architectures"
```

After installing the package, discovery is automatic:

```python
from node_fdm.architectures import get, get_origin

spec = get("my_arch")
origin = get_origin("my_arch")
```

Provider aliases must be unique. Collisions fail explicitly instead of silently
overwriting another installed model.

## 4. Verify the complete chain

Test at least:

1. config selects the public alias;
2. raw fixture data passes through the provider preprocessing hook;
3. the generic resolver returns the expected schema;
4. the generic runtime imports external layers and performs a forward pass;
5. a checkpoint records the provider name, distribution version, normalized
   spec, and spec digest;
6. inference rejects a changed provider/spec instead of loading incompatible
   weights.

Generated datasets, weights, and results should remain in an ignored workspace.
Track only source, configuration, manifests, and small deterministic fixtures.

## 5. Promote only validated models

Research variants stay in their paper repository. Once a model is selected,
stabilized, documented, and compatibility-tested, move the model package—not
the experimental history—into `node-fdm-models`.

Generic behavior belongs in `node-fdm`; validated concrete models belong in
`node-fdm-models`; experiments belong in paper repositories.
