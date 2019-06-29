from typing import List, Union

from pydantic import BaseModel, validator

from pv056_2019.outlier_detection import DETECTORS


class SplitterSchema(BaseModel):
    train_split_dir: str
    test_split_dir: str
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


class ODStepConfigSchema(BaseModel):
    train_split_dir: str
    od_methods: List[OutlierDetectorSchema]
    train_od_dir: str
    n_jobs: int = 1

    @validator("n_jobs")
    def n_jobs_validator(cls, value):
        if value < 1:
            raise ValueError("n_jobs must be greater than 0")

        return value


class RemoveOutliersConfigSchema(BaseModel):
    test_split_dir: str
    train_od_dir: str
    percentage: Union[int, List[int]]
    train_removed_dir: str

    @validator("percentage")
    def percentage_validator(cls, value):
        if value >= 100 or value < 0:
            raise ValueError("Percentage of removed outliers must be between 0 and 100")

        return value


class FilterSchema(BaseModel):
    name: str
    args: List[str] = []


class ClassifierSchema(BaseModel):
    class_name: str
    args: List[str] = []
    filters: List[FilterSchema] = []

    @property
    def name(self):
        return self.class_name.split(".")[-1]


class RunClassifiersCongfigSchema(BaseModel):
    output_folder: str
    weka_jar_path: str
    classifiers: List[ClassifierSchema]
    n_jobs: int = 1

    @validator("n_jobs")
    def n_jobs_validator(cls, value):
        if value < 1:
            raise ValueError("n_jobs must be greater than 0")

        return value
