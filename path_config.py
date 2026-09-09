"""Load PMOF_CODE_DIR and DATA_BASE_DIR for the current workstation.

Only edit ``path_config.yaml`` (in particular ``current_workstation``) to
switch machines; every notebook imports the resulting paths from here.
"""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "path_config.yaml"


def _load_paths() -> tuple[Path, Path]:
    with open(_CONFIG_PATH) as config_file:
        config = yaml.safe_load(config_file)

    workstation = config["current_workstation"]
    try:
        workstation_config = config["workstations"][workstation]
    except KeyError as exc:
        raise ValueError(
            f"Unknown workstation '{workstation}' in {_CONFIG_PATH}"
        ) from exc

    pmof_code_dir = Path(workstation_config["pmof_code_dir"])
    data_base_dir = Path(workstation_config["data_base_dir"])
    data_actioncls_dir = Path(workstation_config["data_actioncls_dir"])
    return pmof_code_dir, data_base_dir, data_actioncls_dir


PMOF_CODE_DIR, DATA_BASE_DIR, DATA_ACTIONCLS_DIR = _load_paths()
