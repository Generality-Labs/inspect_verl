"""MiMo's `opensource-code` environments as an Inspect task.

Mirrors the environment that trained MiMo-V2.6 (mimoagent
`environments/datasets/opensource_code.py`, branch `mimo-oss`): the agent works in
the task's image on a repository at `cwd`; the hidden tests and the verifier script
ship in `test_patch` and exist only at grading time. Reward contract, in order:

1. reset every path the test patch touches to the base commit captured at setup
   (restore it if the base has it, delete it if not: the agent may have created a
   file the patch adds, and `git apply` would then refuse it)
2. `git apply` the test patch
3. run `test_command`; reward is 1 iff it exits 0

and the touched paths are reset again afterwards. A failure in 1 or 2 is reward 0
with `error_category: testbed_corrupted`, as in theirs, so a broken testbed can be
told apart from a wrong answer.

Their setup aborts when the image's git history reaches past the base commit (the
agent could read the fix from `git log`). Here setup records it instead, with a
check that the hidden tests are not already on disk, so `mimo_code_probes` can report
both across many images; the images' build-time guarantees are claims to test.
"""

import hashlib
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import Task, task
from inspect_ai.agent import as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import bash, text_editor
from inspect_ai.util import sandbox

from ._data import MIMO_REPO, mimo_image, read_rows

SOURCE = f"{MIMO_REPO}/code.parquet"
PATCH_IN_BOX = "/tmp/_inspect_verl_test.patch"
# their pod limits (verl config/agent/code/mini-mimocode.yaml)
CPUS, MEMORY = 4, "8g"

_STORE = "inspect_verl"


def patch_paths(patch: str) -> list[str]:
    """Every path a git patch touches, both sides of each `diff --git` header.

    Both sides because for a rename the source must also be reset, or re-applying the
    patch fails on the missing original (their `get_patch_touched_files`).
    """
    files: list[str] = []
    for line in patch.split("\n"):
        match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if match:
            for path in match.groups():
                if path not in files:
                    files.append(path)
    return files


def reset_script(test_patch: str, base_ref: str) -> str | None:
    """Bash returning each patch-touched path to the base commit, or None if none."""
    touched = patch_paths(test_patch)
    if not touched or not base_ref:
        return None
    return (
        'while IFS= read -r tf; do\n  [ -z "$tf" ] && continue\n'
        f'  if git cat-file -e {base_ref}:"$tf" 2>/dev/null; then\n'
        f'    git checkout {base_ref} -- "$tf" 2>/dev/null || true\n'
        '  else\n    git rm -f --cached "$tf" >/dev/null 2>&1 || true\n    rm -f "$tf"\n  fi\n'
        "done <<'EOF_RESET'\n" + "\n".join(touched) + "\nEOF_RESET\n"
    )


def verifier_path(test_command: str) -> str | None:
    """The script a `test_command` runs, e.g. `bash /testbed/mimo_test_command.sh`."""
    for token in shlex.split(test_command):
        if token.startswith("/"):
            return token
    return None


def compose_file(image: str, cwd: str, network: str, platform: str | None = None) -> str:
    """A compose file running `image` idle at `cwd`, cached per configuration.

    Only keys inspect_k8s_sandbox's converter accepts, so the same file runs locally
    and on Hawk: no `platform` unless asked for (the converter refuses it; the images
    are amd64 and so are Hawk's nodes), and no `init`, which Kubernetes ignores and the
    training pods did not have.
    """
    key = hashlib.sha256(f"{image}|{cwd}|{network}|{platform}".encode()).hexdigest()[:16]
    # named compose.yaml: Inspect and inspect_k8s_sandbox recognise a compose file by
    # its name, and on Hawk anything else is passed to Helm as raw values
    path = Path(tempfile.gettempdir()) / "inspect_verl" / key / "compose.yaml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        service: dict[str, Any] = {
            "image": image,
            "working_dir": cwd,
            "command": ["tail", "-f", "/dev/null"],
            "cpus": CPUS,
            "mem_limit": MEMORY,
            "network_mode": network,
        }
        if platform:
            service["platform"] = platform
        path.write_text(yaml.safe_dump({"services": {"default": service}}))
    return str(path)


def samples(
    source: str,
    instance_ids: list[str] | None,
    limit: int | None,
    network: str,
    platform: str | None = None,
) -> list[Sample]:
    rows = read_rows(source)
    if instance_ids is not None:
        wanted = set(instance_ids)
        rows = [r for r in rows if r["instance_id"] in wanted]
    rows = rows[:limit] if limit is not None else rows
    return [
        Sample(
            id=row["instance_id"],
            input=row["problem_statement"],  # what their agent was given, verbatim
            sandbox=(
                "docker",
                compose_file(mimo_image(row["docker_image"]), row["cwd"], network, platform),
            ),
            # hidden from the agent: metadata never reaches the sandbox
            metadata={
                key: row[key]
                for key in ("cwd", "test_patch", "test_command", "verifier_timeout_sec")
            },
        )
        for row in rows
    ]


async def _run(cmd: str, cwd: str, timeout: int | None = None) -> tuple[bool, str]:
    result = await sandbox().exec(["bash", "-lc", cmd], cwd=cwd, timeout=timeout)
    return result.success, (result.stdout + result.stderr)


@solver
def prepare_repo() -> Solver:
    """Setup: a git work tree at `cwd`, its base commit, and the leak checks."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        cwd = meta["cwd"]
        await _run(f"git config --global --add safe.directory {shlex.quote(cwd)}", cwd)
        is_repo, _ = await _run("git rev-parse --git-dir", cwd)
        if not is_repo:  # a bare source tree: its baseline commit is the base
            ok, out = await _run(
                "git init -q && git add -A && git commit -q -m baseline --allow-empty", cwd
            )
            if not ok:
                raise RuntimeError(f"could not create a git work tree at {cwd}: {out[-500:]}")
        ok, out = await _run("git rev-parse HEAD", cwd)
        base = out.strip().split()[-1] if ok and out.strip() else ""
        if len(base) != 40:
            raise RuntimeError(f"could not resolve HEAD at {cwd}: {out[-500:]}")

        _, beyond = await _run(f"git rev-list --all --not {base} | head -n 5", cwd)
        verifier = verifier_path(meta["test_command"])
        verifier_present = verifier is not None and (await _run(f"test -e {verifier}", cwd))[0]
        # the hidden tests already in the tree: the patch reverses cleanly. Piped on
        # stdin, so the hidden tests never touch the box's disk before the agent runs
        check = await sandbox().exec(
            ["git", "apply", "--check", "-R", "-"], input=meta["test_patch"], cwd=cwd
        )
        already_applied = check.success

        state.store.set(
            _STORE,
            {
                "base_ref": base,
                "commits_beyond_base": [c for c in beyond.split() if len(c) == 40],
                "verifier_present": verifier_present,
                "tests_present": already_applied,
            },
        )
        return state

    return solve


@scorer(metrics=[accuracy()])
def test_patch_reward() -> Scorer:
    """Their reward: reset, apply the hidden tests, 1 iff the verifier exits 0."""

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        cwd = meta["cwd"]
        base = state.store.get(_STORE, {}).get("base_ref", "")
        reset = reset_script(meta["test_patch"], base)
        try:
            if reset is not None:
                ok, out = await _run(reset, cwd)
                if not ok:
                    return _corrupted("reset_tests_failed", out)
            await sandbox().write_file(PATCH_IN_BOX, meta["test_patch"])
            ok, out = await _run(f"git apply --verbose {PATCH_IN_BOX}", cwd)
            await _run(f"rm -f {PATCH_IN_BOX}", cwd)
            if not ok:
                return _corrupted("apply_test_patch_failed", out)
            result = await sandbox().exec(
                ["bash", "-lc", meta["test_command"]],
                cwd=cwd,
                timeout=int(meta["verifier_timeout_sec"]),
            )
            return Score(
                value=1 if result.returncode == 0 else 0,
                explanation=(result.stdout + result.stderr)[-4000:],
                metadata={"verifier_returncode": result.returncode},
            )
        finally:
            # leave the touched paths as found, so a rescore starts from the same state
            if reset is not None:
                await _run(reset, cwd)

    return score


def _corrupted(error: str, output: str) -> Score:
    return Score(
        value=0,
        explanation=output[-4000:],
        metadata={"error_category": "testbed_corrupted", "error": error},
    )


@scorer(metrics=[accuracy()])
def history_clean() -> Scorer:
    """1 when no commit past the base is reachable (no fix to read from `git log`)."""

    async def score(state: TaskState, target: Target) -> Score:
        beyond = state.store.get(_STORE, {}).get("commits_beyond_base", [])
        return Score(
            value=0 if beyond else 1,
            explanation=f"reachable commits past base: {beyond}" if beyond else "truncated",
        )

    return score


@scorer(metrics=[accuracy()])
def tests_hidden() -> Scorer:
    """1 when neither the hidden tests nor the verifier script exist before grading."""

    async def score(state: TaskState, target: Target) -> Score:
        record = state.store.get(_STORE, {})
        found = [
            name
            for name, present in (
                ("hidden tests already applied", record.get("tests_present")),
                ("verifier script on disk", record.get("verifier_present")),
            )
            if present
        ]
        return Score(value=0 if found else 1, explanation="; ".join(found) or "hidden")

    return score


def _agent() -> Solver:
    # their agent: bash plus file read/write/edit; swap for any Inspect agent
    return as_solver(react(tools=[bash(timeout=300), text_editor()]))


@task
def mimo_code(
    source: str = SOURCE,
    instance_ids: list[str] | None = None,
    limit: int | None = None,
    network: str = "none",
    platform: str | None = None,
) -> Task:
    """MiMo-V2.6 RL code environments, graded with their reward.

    Args:
        source: The verl parquet: a local path or `<hf dataset repo>/<file>`.
        instance_ids: Run only these tasks.
        limit: Run at most this many (after `instance_ids`).
        network: The container's network mode; none, as the training pods had no egress.
        platform: Force an image platform, e.g. linux/amd64 on an arm64 laptop; unset
            otherwise (Hawk's converter refuses the key).
    """
    return Task(
        dataset=samples(source, instance_ids, limit, network, platform),
        setup=prepare_repo(),
        solver=_agent(),
        scorer=test_patch_reward(),
    )


@task
def mimo_code_probes(
    source: str = SOURCE,
    instance_ids: list[str] | None = None,
    limit: int | None = None,
    network: str = "none",
    platform: str | None = None,
) -> Task:
    """The environments with no agent at all: checks that need no model.

    `test_patch_reward` is the null-agent probe (the untouched repository should
    score 0, or the verifier is vacuous); `history_clean` and `tests_hidden` test the
    image guarantees the training environment assumed but did not check.
    """
    return Task(
        dataset=samples(source, instance_ids, limit, network, platform),
        setup=prepare_repo(),
        solver=[],
        scorer=[test_patch_reward(), history_clean(), tests_hidden()],
    )
