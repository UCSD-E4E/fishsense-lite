import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import psycopg
from fishsense_common.scheduling.arguments import argument
from fishsense_common.scheduling.job_definition import JobDefinition
from fishsense_common.scheduling.ray_job import RayJob
from pyfishsensedev.image.pdf import Pdf
from pyfishsensedev.plane_detector.slate_detector import SlateDetector
from skimage.util import img_as_ubyte

from fishsense_lite.pipeline.tasks.process_raw import process_raw
from fishsense_lite.utils import PSqlConnectionString, parse_psql_connection_string


def get_slate_names(connection_string: PSqlConnectionString) -> Dict[str, str]:
    with psycopg.connect(
        host=connection_string.host,
        port=connection_string.port,
        dbname=connection_string.dbname,
        user=connection_string.username,
        password=connection_string.password,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name, scan_path FROM dive_slates")

            slates_from_database = {row[0]: Path(row[1]) for row in cur.fetchall()}

    # TODO: Hardcoded return value
    base_path = Path("/mnt/fishsense_data/REEF/data/2024.06.20.REEF/new v-slate photos")
    slates_from_database["V1"] = base_path / "SMILE vslate 1.pdf"
    slates_from_database["V2"] = base_path / "SMILE vslate 2.pdf"
    slates_from_database["V3"] = base_path / "SMILE vslate 3.pdf"
    slates_from_database["V4"] = base_path / "SMILE vslate 4.pdf"

    return slates_from_database


def execute(
    dive: Path, connection_string: PSqlConnectionString
) -> Tuple[Path, Set[str]]:
    possible_slate_names: Set[str] = {}
    slate_names = {n: Pdf(f) for n, f in get_slate_names(connection_string).items()}

    for image_file in dive.glob("*.ORF"):
        img = img_as_ubyte(process_raw(image_file))

        for slate_name, pdf in slate_names.items():
            if slate_name in possible_slate_names:
                continue

            slate_detector = SlateDetector(img, pdf)
            if slate_detector.is_valid():
                possible_slate_names.add(slate_name)

    return dive, possible_slate_names


class DetectSlateForDives(RayJob):
    name = "detect_slate_for_dives"

    @property
    def job_count(self) -> int:
        return len(self.dives)

    @property
    def description(self):
        return "Detect slate for requested dives"

    @property
    @argument("dives", required=True, help="List of paths to the dives")
    def dives(self) -> List[str]:
        return self.__dives

    @dives.setter
    def dives(self, value: List[str]):
        self.__dives = value

    @property
    @argument(
        "output",
        required=True,
        help="The path to store the resulting database.",
    )
    def output_path(self) -> str:
        return self.__output_path

    @output_path.setter
    def output_path(self, value: str):
        self.__output_path = value

    @property
    @argument(
        "psql-connection-string",
        help="The connection string to the Postgres database.",
    )
    def psql_connection_string(self) -> str:
        return self.__psql_connection_string

    @psql_connection_string.setter
    def psql_connection_string(self, value: str):
        self.__psql_connection_string = value

    def __init__(self, job_defintion: JobDefinition):
        self.__dives: List[str] = None
        self.__psql_connection_string: str = None

        super().__init__(job_defintion, execute, vram_mb=615)

    def prologue(self):
        dives = {Path(d) for d in self.dives}
        dives = {d for d in dives if d.exists() and d.is_dir()}

        psql_connection_string = parse_psql_connection_string(
            self.psql_connection_string
        )

        return ((dive, psql_connection_string) for dive in dives)

    def epiloge(self, results: Iterable[Tuple[Path, Set[str]]]):
        dive_results = {dive: slate_names for dive, slate_names in results}

        output = Path(self.output_path) / "slate_names.json"
        with output.open("w") as f:
            json.dump(dive_results, f)
