"""Generate API reference pages for mkdocstrings."""

from pathlib import Path

import mkdocs_gen_files

packages_dir = Path("packages")

for pkg_dir in sorted(packages_dir.iterdir()):
    src_dir = pkg_dir / "src"
    if not src_dir.is_dir():
        continue
    for path in sorted(src_dir.rglob("*.py")):
        module_path = path.relative_to(src_dir).with_suffix("")
        doc_path = path.relative_to(src_dir).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        parts = tuple(module_path.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            identifier = ".".join(parts)
            fd.write(f"::: {identifier}")

        mkdocs_gen_files.set_edit_path(full_doc_path, path)
