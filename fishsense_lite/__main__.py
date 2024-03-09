from bom_common.pluggable_cli import Cli

if __name__ == "__main__":
    cli = Cli("./fishsense_lite/plugins.yaml", prog="fishsense-lite")
    cli()
