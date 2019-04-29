from typing import List

from pydantic import BaseModel, validator

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


class RemoveOutliersConfigSchema(BaseModel):
    test_split_dir: str
    train_od_dir: str
    percentage: int
    train_removed_dir: str

    @validator("percentage")
    def percentage_validator(cls, value):
        if value >= 100 or value < 0:
            raise ValueError("Percentage of removed outliers must be between 0 and 100")

        return value
