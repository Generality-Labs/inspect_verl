import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml
from inspect_ai import eval
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from inspect_verl import mimo_code, mimo_code_probes
from inspect_verl._data import mimo_image, read_rows
from inspect_verl.opensource_code import compose_file, patch_paths, reset_script, verifier_path

RENAME = """diff --git a/old/name.py b/new/name.py
similarity index 90%
rename from old/name.py
rename to new/name.py
diff --git a/tests/test_x.py b/tests/test_x.py
new file mode 100644
"""


def test_patch_paths_takes_both_sides_of_a_rename_once() -> None:
    assert patch_paths(RENAME) == ["old/name.py", "new/name.py", "tests/test_x.py"]


def test_reset_restores_base_paths_and_deletes_added_ones() -> None:
    script = reset_script(RENAME, "a" * 40)
    assert script is not None
    assert f"git cat-file -e {'a' * 40}:" in script and 'rm -f "$tf"' in script
    assert script.endswith("old/name.py\nnew/name.py\ntests/test_x.py\nEOF_RESET\n")
    assert reset_script("no diff here", "a" * 40) is None
    assert reset_script(RENAME, "") is None


def test_verifier_path_and_image_names() -> None:
    assert verifier_path("bash /testbed/mimo_test_command.sh") == "/testbed/mimo_test_command.sh"
    assert verifier_path("make test") is None
    assert mimo_image("format-code-task-001457:latest") == (
        "xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457"
    )
    assert mimo_image("ghcr.io/x/y:z") == "ghcr.io/x/y:z"


def test_compose_runs_the_image_idle_without_network() -> None:
    compose = yaml.safe_load(Path(compose_file("img:tag", "/testbed", "none")).read_text())
    service = compose["services"]["default"]
    assert service["image"] == "img:tag" and service["working_dir"] == "/testbed"
    assert service["network_mode"] == "none" and "platform" not in service


def test_the_compose_converts_for_hawk() -> None:
    """Hawk runs compose sandboxes through inspect_k8s_sandbox's converter, which
    refuses keys it does not know (`platform` did, on the first version)."""
    converter = pytest.importorskip("k8s_sandbox.compose")
    values = converter.convert_compose_to_helm_values(
        Path(compose_file("xiaomimimo/mimo-v2.6-rl-oss:format-code-task-003060", "/testbed", "none"))
    )
    service = values["services"]["default"]
    assert service["image"].endswith("format-code-task-003060")
    assert service["workingDir"] == "/testbed"


# --- end to end on a tiny image: the reward contract itself -----------------------

BUGGY = 'add() { echo $(( $1 - $2 )); }\n'
FIXED = 'add() { echo $(( $1 + $2 )); }\n'
VERIFIER = "cd /testbed && bash tests/test_calc.sh\n"
TEST = 'source /testbed/calc.sh\n[ "$(add 2 3)" = 5 ]\n'


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _test_patch(tmp: Path) -> str:
    """The hidden tests and the verifier, as a diff against the buggy repository."""
    repo = tmp / "patchrepo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "calc.sh").write_text(BUGGY)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    (repo / "tests").mkdir()
    (repo / "tests/test_calc.sh").write_text(TEST)
    (repo / "mimo_test_command.sh").write_text(VERIFIER)
    _git(repo, "add", "-A")
    return _git(repo, "diff", "--cached")


def _image(tmp: Path, tag: str, leaky: bool) -> str:
    context = tmp / tag
    context.mkdir()
    (context / "calc.sh").write_text(BUGGY)
    fix = (
        "&& git checkout -qb fix && printf 'add() { echo $(( $1 + $2 )); }\\n' > calc.sh "
        "&& git commit -qam fix && git checkout -q master "
        if leaky
        else ""
    )
    (context / "Dockerfile").write_text(
        "FROM alpine:3.20\nRUN apk add --no-cache git bash\nWORKDIR /testbed\nCOPY calc.sh .\n"
        "RUN git init -q -b master && git config --global user.email t@t "
        "&& git config --global user.name t && git add -A && git commit -qm base " + fix + "\n"
    )
    image = f"inspect-verl/test:{tag}"  # a slash: not a MiMo tag
    subprocess.run(["docker", "build", "-q", "-t", image, str(context)], check=True)
    return image


def _source(tmp: Path, rows: list[dict]) -> str:
    frame = pd.DataFrame({"extra_info": [{"instance_json": json.dumps(r)} for r in rows]})
    path = tmp / "rows.parquet"
    frame.to_parquet(path)
    return str(path)


def _row(instance_id: str, image: str, patch: str) -> dict:
    return {
        "instance_id": instance_id,
        "docker_image": image,
        "cwd": "/testbed",
        "problem_statement": "add() subtracts",
        "test_patch": patch,
        "test_command": "bash /testbed/mimo_test_command.sh",
        "verifier_timeout_sec": 60,
    }


@solver
def write_file(path: str, content: str, also: dict[str, str] | None = None) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await sandbox().write_file(path, content)
        for extra, text in (also or {}).items():
            await sandbox().exec(["mkdir", "-p", str(Path(extra).parent)])
            await sandbox().write_file(extra, text)
        return state

    return solve


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("verl")
    patch = _test_patch(tmp)
    clean, leaky = _image(tmp, "clean", leaky=False), _image(tmp, "leaky", leaky=True)
    source = _source(tmp, [_row("clean", clean, patch), _row("leaky", leaky, patch)])
    native = subprocess.run(
        ["docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().replace("aarch64", "arm64").replace("x86_64", "amd64")
    return {"source": source, "platform": native, "rows": read_rows(source)}


def _scores(log) -> dict[str, dict[str, object]]:
    return {str(s.id): {k: v.value for k, v in (s.scores or {}).items()} for s in log.samples}


@pytest.mark.docker
def test_doing_nothing_scores_zero_and_leaks_are_reported(world: dict) -> None:
    log = eval(
        mimo_code_probes(source=world["source"], platform=world["platform"]),
        model="mockllm/model", display="none",
    )[0]
    assert log.status == "success", log.error
    scores = _scores(log)
    # null agent: the untouched repository fails the hidden tests
    assert scores["clean"]["test_patch_reward"] == 0
    assert scores["clean"]["history_clean"] == 1 and scores["clean"]["tests_hidden"] == 1
    # the fix is reachable on a branch: an agent could read it from git log
    assert scores["leaky"]["history_clean"] == 0


@pytest.mark.docker
def test_a_real_fix_scores_one(world: dict) -> None:
    log = eval(
        mimo_code(source=world["source"], instance_ids=["clean"], platform=world["platform"]),
        solver=write_file("/testbed/calc.sh", FIXED), model="mockllm/model", display="none",
    )[0]
    assert _scores(log)["clean"]["test_patch_reward"] == 1


@pytest.mark.docker
def test_a_planted_test_file_is_replaced_by_the_hidden_one(world: dict) -> None:
    """An agent that writes its own always-passing test where the hidden test goes
    must still be graded by the hidden test: no reward without the fix."""
    cheat = write_file(
        "/testbed/README", "x",
        also={"/testbed/tests/test_calc.sh": "exit 0\n", "/testbed/mimo_test_command.sh": "exit 0\n"},
    )
    log = eval(
        mimo_code(source=world["source"], instance_ids=["clean"], platform=world["platform"]),
        solver=cheat, model="mockllm/model", display="none",
    )[0]
    score = log.samples[0].scores["test_patch_reward"]  # type: ignore[index]
    assert score.value == 0 and "error_category" not in (score.metadata or {})
