from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".pptx"}

def is_hidden_or_temporary(relative_path: Path) -> bool:
    return (
        any(part.startswith(".") for part in relative_path.parts)
        or relative_path.name.startswith("~$")
        or relative_path.name.endswith(".tmp")
    )


def collect_files(paths: list[str | Path]) -> list[Path]:
    files: set[Path] = set()

    for raw_path in paths:
        root = Path(raw_path).expanduser().resolve()

        if not root.exists():
            raise FileNotFoundError(
                f"Input path does not exist: {root}"
            )

        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = root.rglob("*")
        else:
            continue

        for candidate in candidates:
            if not candidate.is_file():
                continue

            # Inspect paths relative to the requested directory.
            # Do not reject an entire project because an ancestor is hidden.
            relative = (
                candidate.relative_to(root)
                if root.is_dir()
                else Path(candidate.name)
            )

            if is_hidden_or_temporary(relative):
                continue

            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            files.add(candidate.resolve())

    return sorted(files)
