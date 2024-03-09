from bom_common.pluggable_cli import Cli
from wakepy import keep

if __name__ == "__main__":
    with keep.running():
        cli = Cli("./fishsense_lite/plugins.yaml", prog="fishsense-lite")
        cli()
