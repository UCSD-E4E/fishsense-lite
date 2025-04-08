from fishsense_common.pipeline.decorators import task


@task(output_name="segmentation_mask")
def calculate_segmentation_mask(segmentation_model: None) -> None:
    if segmentation_model is None:
        return None

    raise NotImplementedError
