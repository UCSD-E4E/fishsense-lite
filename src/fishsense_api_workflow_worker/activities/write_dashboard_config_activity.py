"""Activity to write dashboard configuration file."""

from pathlib import Path
from typing import Any, Dict, List

import yaml
from temporalio import activity

from fishsense_api_workflow_worker.models.label_studio_project import LabelStudioProject


def __generate_link(
    title: str, description: str, link: str, icon_name: str
) -> Dict[str, Any]:
    return {
        "title": title,
        "description": description,
        "link": link,
        "icon": {"name": icon_name, "wrap": True},
    }


@activity.defn
def write_dashboard_config_activity(
    laser_labeling_projects: List[LabelStudioProject],
    species_labeling_projects: List[LabelStudioProject],
    head_tail_labeling_projects: List[LabelStudioProject],
    slate_labeling_projects: List[LabelStudioProject],
):
    """Activity to write dashboard configuration file."""

    target_config_path = Path("/config/dashboard_config.yaml")
    analytics_base_url = "https://analytics.fishsense.e4e.ucsd.edu/superset/dashboard/"

    dashboard_config: Dict[str, Any] = {
        "title": "E4E FishSense",
        "services": {},
    }

    if laser_labeling_projects:
        labeling_links = []
        for project in laser_labeling_projects:
            labeling_links.append(
                __generate_link(
                    project.name,
                    f"{project.name} labeling project",
                    f"https://labeler.e4e.ucsd.edu/projects/{project.id}",
                    "streamline-sharp:label-folder-tag-remix",
                )
            )

        dashboard_config["services"]["Laser Labeling"] = labeling_links

    if species_labeling_projects:
        labeling_links = []
        for project in species_labeling_projects:
            labeling_links.append(
                __generate_link(
                    project.name,
                    f"{project.name} labeling project",
                    f"https://labeler.e4e.ucsd.edu/projects/{project.id}",
                    "streamline-sharp:label-folder-tag-remix",
                )
            )

        dashboard_config["services"]["Species Labeling"] = labeling_links

    if head_tail_labeling_projects:
        labeling_links = []
        for project in head_tail_labeling_projects:
            labeling_links.append(
                __generate_link(
                    project.name,
                    f"{project.name} labeling project",
                    f"https://labeler.e4e.ucsd.edu/projects/{project.id}",
                    "streamline-sharp:label-folder-tag-remix",
                )
            )

        dashboard_config["services"]["Head/Tail Labeling"] = labeling_links

    if slate_labeling_projects:
        labeling_links = []
        for project in slate_labeling_projects:
            labeling_links.append(
                __generate_link(
                    project.name,
                    f"{project.name} labeling project",
                    f"https://labeler.e4e.ucsd.edu/projects/{project.id}",
                    "streamline-sharp:label-folder-tag-remix",
                )
            )

        dashboard_config["services"]["Slate Labeling"] = labeling_links

    dashboard_config["services"]["Results"] = [
        __generate_link(
            "Lengths",
            "Lengths dashboard from Apache Superset",
            f"{analytics_base_url}reef-smile-lengths",
            "simple-icons:apachesuperset",
        ),
        __generate_link(
            "Metrics",
            "Metrics dashboard from Apache Superset",
            f"{analytics_base_url}reef-smile-metrics",
            "simple-icons:apachesuperset",
        ),
    ]

    dashboard_config["services"]["Administration"] = [
        __generate_link(
            "Workflows",
            "Temporal Workflows",
            "https://workflows.fishsense.e4e.ucsd.edu/",
            "simple-icons:temporal",
        )
    ]

    target_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            dashboard_config,
            f,
            sort_keys=False,
        )
