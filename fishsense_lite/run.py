"""Entry point for the module when running `fsl`"""

from fishsense_common.pluggable_cli import Cli

from fishsense_lite.commands.calibrate_laser import CalibrateLaser
from fishsense_lite.commands.calibrate_lens import CalibrateLens
from fishsense_lite.commands.field_calibrate_laser import FieldCalibrateLaser
from fishsense_lite.commands.preprocess import Preprocess
from fishsense_lite.commands.process import Process


def main():
    """Main entry point for the CLI."""
    cli = Cli(
        name="fsl",
        description="The command line tool for processing data from the FishSense Lite product line.",
    )

    cli.add(CalibrateLaser())
    cli.add(CalibrateLens())
    cli.add(FieldCalibrateLaser())
    cli.add(Preprocess())
    cli.add(Process())

    cli()


if __name__ == "__main__":
    main()
