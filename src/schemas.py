from pydantic import BaseModel, validator
from typing import List, Dict, Union
from outlier_detection import DETECTORS


class OutlierDataSchema(BaseModel):
    data_paths: List[str]
    detectors: Dict[str, Dict[str, Union[str, int, float]]] = {
        x: {} for x in DETECTORS.keys()
    }
    output_dir: str

    @validator("detectors", whole=True)
    def supported_detectors(cls, values):
        for key in values.keys():
            if key not in DETECTORS.keys():
                raise ValueError(
                    "Detector {} is not supported. Supported detectors are: {}".format(
                        key, ", ".join(DETECTORS.keys())
                    )
                )

        return values
