# node-fdm Documentation

`node-fdm` is a physics-guided Neural ODE framework for vertical flight dynamics. It supports multiple architectures (e.g., `opensky_2025`, `qar`, or your own) sharing the same training/inference tooling.

Use the OpenSky 2025 pipeline as a reference example: build an aircraft list, download Mode S data, preprocess + enrich with ERA5, train Neural ODEs per aircraft type, and benchmark against BADA. You can swap in any registered architecture as long as your data and preprocessing match its columns.

Use this site as a quick reference:
- **Guide**: install the project and run pipelines (OpenSky 2025 as example).
- **How-to**: generic recipes for training, inference, configuration, and creating architectures.
- **Concepts**: model inputs/outputs, column groups, and how architectures are structured.
- **Reference API**: auto-generated mkdocstrings for the package (multi-architecture).

If you only need a fast start, jump to `Guide → Quickstart`.
