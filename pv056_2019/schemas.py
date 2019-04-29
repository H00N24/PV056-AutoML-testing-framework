from pydantic import BaseModel, validator
from typing import List, Dict, Union
from pv056_2019.outlier_detection import DETECTORS


class TrainTestSplitSchema(BaseModel):
    train_split_dir: str
    test_split_dir: str


class SplitterSchema(TrainTestSplitSchema):
    data_path: str


class OutlierDetectorSchema(BaseModel):
    name: str
    parameters: dict

    @validator("name")
    def detector_name(cls, value):
        if value not in DETECTORS.keys():
            raise ValueError(
                "Detector {} is not supported. Supported detectors are: {}".format(
                    value, ", ".join(DETECTORS.keys())
                )
            )

        return value


class ODStepConfigSchema(TrainTestSplitSchema):
    od_methods: List[OutlierDetectorSchema]
    train_od_dir: str


class OutlierDataSchema(BaseModel):
    data_paths: List[str]
    output_dir: str
    detectors: Dict[str, Dict[str, Union[float, str]]] = {
        x: {} for x in DETECTORS.keys()
    }

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
