import subprocess
import sys


def test_package_importable_without_settings_in_cwd(tmp_path):
    """Regression for finding #2: importing the package, including server.py
    and the controllers, must not require a settings.toml in the current
    working directory. PG connection details are read lazily at lifespan
    startup, not at import time."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import fishsense_api.server\n"
                "from fishsense_api.controllers import (\n"
                "    camera_controller, dive_controller, dive_slate_controller,\n"
                "    fish_controller, image_controller, label_controller,\n"
                "    user_controller,\n"
                ")\n"
                "print(fishsense_api.server.app.title)\n"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"import failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "FastAPI" in result.stdout
