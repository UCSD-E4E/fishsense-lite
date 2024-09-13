"""Entry point for the module when running `fsl`"""

from bom_common.pluggable_cli import Cli
from wakepy import keep


def main():
    """Main entry point for the CLI."""
    # with keep.running():
    cli = Cli("./fishsense_lite/plugins.yaml", prog="fishsense-lite")
    cli()


if __name__ == "__main__":
    main()
