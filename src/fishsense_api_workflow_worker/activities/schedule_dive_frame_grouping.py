from pathlib import Path
from typing import Iterable

from temporalio import activity
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleSpec,
    ScheduleState,
    TLSConfig,
)

from fishsense_api_workflow_worker.models.dive import Dive
from fishsense_api_workflow_worker.models.image import Image

DATA_WORKER_TASK_QUEUE_NAME = "fishsense_data_queue"


@activity.defn
async def schedule_dive_frame_grouping(
    dive: Dive,
    images: Iterable[Image],
    temporal_host: str,
    temporal_port: int,
    temporal_tls: bool = False,
    temporal_client_cert: str | None = None,
    temporal_client_private_key: str | None = None,
    temporal_server_root_ca_cert: str | None = None,
    temporal_domain: str | None = None,
):
    """Schedule the dive frame grouping activity."""
    log = activity.logger

    log.info("Scheduling workflow for dive: %s.", dive.name)

    tls_config: TLSConfig | None = None
    if temporal_tls:
        assert temporal_client_cert is not None
        assert temporal_client_private_key is not None

        with open(temporal_client_cert, "rb") as f:
            client_cert = f.read()
        with open(temporal_client_private_key, "rb") as f:
            client_private_key = f.read()

        server_root_ca_cert: bytes | None = None
        if temporal_server_root_ca_cert:
            with open(temporal_server_root_ca_cert, "rb") as f:
                server_root_ca_cert = f.read()

        tls_config = TLSConfig(
            client_cert=client_cert,
            client_private_key=client_private_key,
            server_root_ca_cert=server_root_ca_cert,
            domain=temporal_domain,
        )

    client = await Client.connect(f"{temporal_host}:{temporal_port}", tls=tls_config)

    await client.execute_workflow(
        "dive_frame_grouping_workflow",
        args=(dive, images),
        id=f"dive-frame-grouping-{dive.id}",
        task_queue=DATA_WORKER_TASK_QUEUE_NAME,
        retry_policy=None,
    )
