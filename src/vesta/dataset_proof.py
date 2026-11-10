"""Whether a dataset file is ours, proved by its hash in a signed `SHA256SUMS`
(signed with ed25519).

Each dataset release ships `SHA256SUMS` (one line per file, `<sha256>  <path>`) and its
signature `SHA256SUMS.sig`, made by the maintainers with `ssh-keygen -Y sign -n 11footy-dataset`.
The public key is in `config/vesta/dataset-signers` (OpenSSH's allowed-signers format, principal
`11footy-dataset`). A file is proved ours only when:

1. its header names our dataset ('11footy');
2. a `SHA256SUMS` in its folder or one of its parents lists its hash under its name;
3. that `SHA256SUMS`'s signature verifies against the shipped key (`ssh-keygen -Y verify`, which
   ships with Windows, macOS and Linux; no Python dependency).

Anything short of that is not proof: the header alone is a claim anyone can write, and a hash list
beside the data proves the download intact, not whose it is. A file not proved is read as a
foreign dataset (refit, `not_validated_on_dataset`).
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from vesta.data_format import OUR_DATASET
from vesta.datasets import DatasetFile

ROOT = Path(__file__).resolve().parents[2]
SIGNERS = ROOT / "config/vesta/dataset-signers"
PRINCIPAL = NAMESPACE = "11footy-dataset"
SUMS = "SHA256SUMS"
PARENTS = 3  # how far up from a file its SHA256SUMS is looked for
# the desktop window runs without a console: on Windows, no console flashes per file checked
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class Proof:
    ours: bool
    level: str  # "signed", "listed" (hash listed, signature missing or not verified), "none"
    reason: str


def _find_sums_file(path: Path) -> Path | None:
    for folder in [path.parent, *path.parent.parents][:PARENTS + 1]:
        if (folder / SUMS).is_file():
            return folder / SUMS
    return None


def _is_listed(sums: Path, item: DatasetFile) -> bool:
    for line in sums.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() == item.sha256 \
                and Path(parts[1].lstrip("*")).name == item.path.name:  # fmt: skip
            return True
    return False


def verify_signature(sums: Path, signers: Path = SIGNERS) -> tuple[bool, str]:
    signature = sums.with_name(SUMS + ".sig")
    if not signature.is_file():
        return False, f"no {signature.name} beside {sums}"
    if not signers.is_file() or not any(
            line.strip() and not line.startswith("#")
            for line in signers.read_text(encoding="utf-8").splitlines()):  # fmt: skip
        return False, f"no signing key in {signers}"
    program = shutil.which("ssh-keygen")
    if program is None:
        return False, "ssh-keygen not found: the signature cannot be checked"
    with open(sums, "rb") as handle:
        done = subprocess.run([program, "-Y", "verify", "-f", str(signers), "-I", PRINCIPAL,
                               "-n", NAMESPACE, "-s", str(signature)], stdin=handle,
                              capture_output=True, timeout=60, check=False,
                              creationflags=NO_WINDOW)  # fmt: skip
    if done.returncode != 0:
        return False, f"the signature of {sums} does not verify"
    return True, "signed"


def prove(item: DatasetFile, signers: Path = SIGNERS) -> Proof:
    if item.header.dataset != OUR_DATASET:
        return Proof(False, "none", f"dataset '{item.header.dataset}'")
    sums = _find_sums_file(item.path)
    if sums is None:
        return Proof(False, "none", f"no {SUMS} beside {item.path.name}")
    if not _is_listed(sums, item):
        return Proof(False, "none", f"{item.path.name}'s hash is not in {sums}")
    verified, reason = verify_signature(sums, signers)
    if not verified:
        return Proof(False, "listed", reason)
    return Proof(True, "signed", f"{item.path.name} listed in the signed {sums.name}")
