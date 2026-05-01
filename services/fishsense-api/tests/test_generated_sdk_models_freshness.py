"""Phase 7 regression guard: committed `_generated.py` is up-to-date.

The fishsense-api-sdk's wire-format models are generated from this
service's OpenAPI schema by `tools/generate_sdk_models.py`. The
generated output is committed to the repo so consumers don't need a
build step. This test re-runs generation and compares to the committed
file — any schema-affecting change to the API must be paired with a
regeneration commit, or this test fails.

Failure remediation: from the repo root, run

    uv run --package fishsense-api python tools/generate_sdk_models.py

and commit the resulting diff.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_OUTPUT = (
    REPO_ROOT
    / "libs"
    / "fishsense-api-sdk"
    / "src"
    / "fishsense_api_sdk"
    / "models"
    / "_generated.py"
)


def _dump_openapi_schema() -> dict:
    # pylint: disable=import-outside-toplevel,unused-import
    # Controllers register routes via @app.get/.put/etc. side effects on
    # import; they MUST be imported after the env-var seed below so dynaconf's
    # eager validation has placeholders to find.
    os.environ.setdefault("E4EFS_POSTGRES__HOST", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__PORT", "5432")
    os.environ.setdefault("E4EFS_POSTGRES__USERNAME", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__PASSWORD", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__DATABASE", "ignored")

    import fishsense_api.controllers.camera_controller  # noqa: F401
    import fishsense_api.controllers.dive_controller  # noqa: F401
    import fishsense_api.controllers.dive_slate_controller  # noqa: F401
    import fishsense_api.controllers.fish_controller  # noqa: F401
    import fishsense_api.controllers.image_controller  # noqa: F401
    import fishsense_api.controllers.label_controller  # noqa: F401
    import fishsense_api.controllers.user_controller  # noqa: F401
    from fishsense_api.server import app

    return app.openapi()


def test_generated_sdk_models_match_current_schema():
    schema = _dump_openapi_schema()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        schema_path = tmp / "openapi.json"
        output_path = tmp / "generated.py"

        schema_path.write_text(json.dumps(schema))

        subprocess.run(
            [
                "datamodel-codegen",
                "--input",
                str(schema_path),
                "--input-file-type",
                "openapi",
                "--output",
                str(output_path),
                "--output-model-type",
                "pydantic_v2.BaseModel",
                "--use-standard-collections",
                "--use-union-operator",
                "--use-schema-description",
                "--disable-timestamp",
            ],
            check=True,
        )

        regenerated = output_path.read_text()

    committed = GENERATED_OUTPUT.read_text()
    assert regenerated == committed, (
        "libs/fishsense-api-sdk/src/fishsense_api_sdk/models/_generated.py is "
        "stale relative to the current OpenAPI schema. Regenerate via:\n"
        "  uv run --package fishsense-api python tools/generate_sdk_models.py\n"
        "and commit the resulting diff."
    )
