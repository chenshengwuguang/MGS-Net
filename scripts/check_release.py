"""Audit an explicit text-source manifest, current tracked files and Git history."""

import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SPECIAL = {"LICENSE", "MANIFEST.in", ".gitignore", ".gitattributes"}
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml"}
MACHINE_PATH = re.compile(r"(?:(?<![A-Za-z0-9_])[A-Za-z]:[\\/]|/(?:home|Users)/)")
KEY_MATERIAL = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}"
)


def check_text(name, payload):
    suffix = PurePosixPath(name).suffix
    if name not in SPECIAL and name != "release-files.txt" and suffix not in SUFFIXES:
        raise ValueError(f"File type outside the source-only manifest: {name}")
    text = payload.decode("utf-8")
    if "\x00" in text or MACHINE_PATH.search(text) or KEY_MATERIAL.search(text):
        raise ValueError(
            f"Unexpected binary content, machine path or credential in {name}"
        )


def git(*args):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True
    ).stdout


def main():
    entries = (ROOT / "release-files.txt").read_text(encoding="utf-8").splitlines()
    allowed = set(entries)
    if not entries or len(allowed) != len(entries) or entries != sorted(entries):
        raise ValueError(
            "Source manifest must be a nonempty sorted list without duplicates."
        )
    for name in entries:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Manifest paths must be safe, relative POSIX paths.")
        check_text(name, (ROOT / name).read_bytes())
    if (ROOT / ".git").exists():
        tracked = set(git("ls-files", "-z").decode().rstrip("\x00").split("\x00")) - {
            ""
        }
        if tracked != allowed:
            raise ValueError(
                f"Tracked files differ from manifest; extra={sorted(tracked - allowed)}, missing={sorted(allowed - tracked)}"
            )
        # Inspect every historical tree, including removed files and prior contents.
        try:
            revisions = git("rev-list", "--all").decode().splitlines()
        except subprocess.CalledProcessError:
            revisions = []
        seen = set()
        for revision in revisions:
            for entry in git("ls-tree", "-r", "-z", revision).split(b"\x00"):
                if not entry:
                    continue
                meta, name = entry.split(b"\t", 1)
                name = name.decode("utf-8")
                mode, kind, object_id = meta.split()
                if (
                    name not in allowed
                    or kind != b"blob"
                    or mode not in {b"100644", b"100755"}
                ):
                    raise ValueError(f"Unexpected historical file: {name}")
                if object_id not in seen:
                    check_text(name, git("cat-file", "blob", object_id.decode()))
                    seen.add(object_id)
    print(f"Source-only audit passed: {len(allowed)} approved text files.")


if __name__ == "__main__":
    main()
