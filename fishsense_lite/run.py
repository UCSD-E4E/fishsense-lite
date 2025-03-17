"""Entry point for the module when running `fsl`"""

from fishsense_common.scheduling.cli_scheduler import CliScheduler

from fishsense_lite.jobs.preprocess import Preprocess
from fishsense_lite.jobs.preprocess_with_laser import PreprocessWithLaser

# from fishsense_lite.commands.calibrate_laser import CalibrateLaser
# from fishsense_lite.commands.calibrate_lens import CalibrateLens
# from fishsense_lite.commands.field_calibrate_laser import FieldCalibrateLaser
# from fishsense_lite.commands.preprocess import Preprocess
# from fishsense_lite.commands.process import Process

# from fishsense_lite.commands.label_studio import LabelStudioCommand


def main():
    """Main entry point for the CLI."""
    cli = CliScheduler(
        name="fsl",
        description="The command line tool for processing data from the FishSense Lite product line.",
    )

    cli.register_job_type(Preprocess)
    cli.register_job_type(PreprocessWithLaser)

    # cli.add(CalibrateLaser())
    # cli.add(CalibrateLens())
    # cli.add(FieldCalibrateLaser())
    # cli.add(Process())
    # cli.add(LabelStudioCommand())

    cli()


if __name__ == "__main__":
    main()
