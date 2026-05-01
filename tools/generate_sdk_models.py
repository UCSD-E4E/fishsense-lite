# pylint: disable=import-outside-toplevel,unused-import
"""Regenerate the SDK's wire-format models from fishsense-api's OpenAPI schema.

Owns the pipeline:
  1. Boot the FastAPI app in-process (no uvicorn) and dump `app.openapi()`.
  2. Run `datamodel-codegen` over that schema.
  3. Write the result to
     `libs/fishsense-api-sdk/src/fishsense_api_sdk/models/_generated.py`.

This is the source of truth for SDK shape going forward. The hand-written
`models/*.py` files are a transitional layer the test_sdk_drift comparison
will check against `_generated.py` once Phase 7 rolls fully forward; for
now both coexist so swapping consumers can happen incrementally.

Run via:

    uv run --package fishsense-api python tools/generate_sdk_models.py

The fishsense-api dynaconf settings are populated with placeholder
values — generation never touches a real database.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATED_OUTPUT = (
    REPO_ROOT
    / "libs"
    / "fishsense-api-sdk"
    / "src"
    / "fishsense_api_sdk"
    / "models"
    / "_generated.py"
)


def dump_openapi_schema() -> dict:
    """Import the FastAPI app and dump its OpenAPI schema.

    Controllers register routes at import time via the module-level
    `@app.get(...)` decorators, so we explicitly import every controller
    module before calling `app.openapi()` to make sure the schema is
    complete.
    """
    # Placeholder DB env so dynaconf validators don't reject the import.
    # No real connection is made — `setup_database()` runs in `lifespan`,
    # not at import time.
    os.environ.setdefault("E4EFS_POSTGRES__HOST", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__PORT", "5432")
    os.environ.setdefault("E4EFS_POSTGRES__USERNAME", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__PASSWORD", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__DATABASE", "ignored")

    # Importing controllers registers routes on `app`.
    import fishsense_api.controllers.camera_controller  # noqa: F401
    import fishsense_api.controllers.dive_controller  # noqa: F401
    import fishsense_api.controllers.dive_slate_controller  # noqa: F401
    import fishsense_api.controllers.fish_controller  # noqa: F401
    import fishsense_api.controllers.image_controller  # noqa: F401
    import fishsense_api.controllers.label_controller  # noqa: F401
    import fishsense_api.controllers.user_controller  # noqa: F401
    from fishsense_api.server import app

    return app.openapi()


def run_codegen(schema: dict) -> str:
    """Run `datamodel-codegen` over `schema` and return the generated source."""
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

        return output_path.read_text()


def main() -> int:
    schema = dump_openapi_schema()
    generated = run_codegen(schema)
    GENERATED_OUTPUT.write_text(generated)
    print(f"wrote {GENERATED_OUTPUT.relative_to(REPO_ROOT)} ({len(generated)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
