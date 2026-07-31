# node-fdm-models

Official, validated architecture specifications for `node-fdm`.

`node-fdm` contains the generic runtime and extension contracts. This package
contains concrete reusable models that have passed validation. Research variants
remain in their paper repositories until promotion.

Third-party packages can expose their own catalog through the
`node_fdm.architectures` entry-point group; users then select an architecture by
its declared alias without cloning or modifying either library.
