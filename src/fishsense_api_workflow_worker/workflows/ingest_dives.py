from datetime import timedelta
from typing import Iterable

from temporalio import workflow

from fishsense_api_workflow_worker.models.dive import Dive
from fishsense_api_workflow_worker.models.image import Image


@workflow.defn
class IngestDivesWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(
        self,
        database_url: str,
        temporal_host: str,
        temporal_port: int,
        temporal_tls: bool = False,
        temporal_client_cert: str | None = None,
        temporal_client_private_key: str | None = None,
        temporal_server_root_ca_cert: str | None = None,
        temporal_domain: str | None = None,
    ):
        dive: Dive = None
        images: Iterable[Image] = None

        dive, images = await workflow.execute_activity(
            "select_dives",
            args=(database_url,),
            schedule_to_close_timeout=timedelta(minutes=10),
        )
        if dive is not None and images is not None:
            await workflow.execute_activity(
                "schedule_dive_frame_grouping",
                args=(
                    dive,
                    images,
                    temporal_host,
                    temporal_port,
                    temporal_tls,
                    temporal_client_cert,
                    temporal_client_private_key,
                    temporal_server_root_ca_cert,
                    temporal_domain,
                ),
                schedule_to_close_timeout=timedelta(minutes=10),
            )
        else:
            workflow.logger.info("No dives to process.")
