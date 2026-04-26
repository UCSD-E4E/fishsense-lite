"""Entry point for the FishSense API Workflow Worker."""

import asyncio

from fishsense_api_workflow_worker.worker import main

if __name__ == "__main__":
    asyncio.run(main())
