from __future__ import absolute_import

import os
import warnings
import re
from typing import Any, Dict, List, Optional

import arff
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from pv056_2019.outlier_detection import DETECTORS
from pv056_2019.utils import ID_NAME, OD_VALUE_NAME
from pv056_2019.schemas import OutlierDetectorSchema

warnings.simplefilter(action="ignore", category=UserWarning)


class DataFrameArff(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        arff_data: Optional[dict] = kwargs.pop("arff_data", None)
        if arff_data is None:
            super().__init__(*args, **kwargs)
        else:
            columns = [x[0] for x in arff_data["attributes"]]

            super().__init__(arff_data["data"], columns=columns, **kwargs)

            self._arff_data: Dict[str, Any] = {}
            for key, item in arff_data.items():
                if key.lower() != "data":
                    self._arff_data.update({key: item})
            self._arff_data["relation"] = self._arff_data["relation"].replace("_", "-")

    def arff_data(self):
        data = self._arff_data
        data.update({"data": self.replace(np.nan, None).values.tolist()})
        return data

    def arff_dumps(self):
        return arff.dumps(self.arff_data())

    def arff_dump(self, file_path: str):
        with open(file_path, "w") as output_file:
            data = self._arff_data
            data.update({"data": self.replace(np.nan, None).values.tolist()})
            arff.dump(data, output_file)

    def _binarize_categorical_values(self) -> pd.DataFrame:
        encoded_dataframe = pd.DataFrame()
        for attr, values in self._arff_data["attributes"][:-1]:
            enc = OneHotEncoder(handle_unknown="ignore")
            enc.fit(np.array(values).reshape(-1, 1))
            if isinstance(values, list):
                transformed_data = enc.transform(
                    self[attr].values.reshape(-1, 1)
                ).toarray()
                columns_index = pd.MultiIndex.from_product(
                    [[attr], values], names=["0", "1"]
                )
            elif values.lower() in {"numeric", "real", "integer"}:
                imputer = SimpleImputer(
                    missing_values=np.nan, strategy="mean"
                )  # XXX settings

                real_values = self[attr].values.astype(float)

                if not np.isnan(real_values).all():
                    transformed_data = imputer.fit_transform(
                        real_values.reshape(-1, 1)
                    ).reshape(-1, 1)
                else:
                    transformed_data = np.zeros(real_values.reshape(-1, 1).shape)
                columns_index = pd.MultiIndex.from_product(
                    [[attr], [values]], names=["0", "1"]
                )
            elif values.lower() == "string":
                imputer = SimpleImputer(missing_values=None, strategy="most_frequent")
                imp_data = imputer.fit_transform(
                    self[attr].values.reshape(-1, 1)
                ).reshape(-1, 1)
                transformed_data = enc.transform(imp_data).toarray()
                columns_index = pd.MultiIndex.from_product(
                    [[attr], range(transformed_data.shape[1])], names=["0", "1"]
                )
            else:
                raise ValueError(attr, values)

            new = pd.DataFrame(transformed_data, columns=columns_index)
            if encoded_dataframe.empty:
                encoded_dataframe = new
            else:
                encoded_dataframe = encoded_dataframe.join(new)

        return encoded_dataframe

    def add_index_column(self):
        if ID_NAME not in self.columns:
            self.insert(loc=0, column=ID_NAME, value=self.index.values)
            self._arff_data["attributes"].insert(0, (ID_NAME, "NUMERIC"))
        return self

    def apply_outlier_detector(self, detector_schema: OutlierDetectorSchema):
        detector = DETECTORS[detector_schema.name](**detector_schema.parameters)

        dataframe_without_id = DataFrameArff(
            self.loc[:, self.columns != ID_NAME].values,
            columns=self.columns[self.columns != ID_NAME],
        )
        dataframe_without_id._arff_data = {
            **self._arff_data,
            "attributes": [x for x in self._arff_data["attributes"] if x[0] != ID_NAME],
        }

        detector.compute_scores(dataframe_without_id, self[self.columns[-1]])

        new_frame = DataFrameArff(self.values, columns=self.columns)
        new_frame._arff_data = self._arff_data

        new_frame.insert(
            loc=len(self.columns) - 1, column=OD_VALUE_NAME, value=detector.values
        )
        new_frame._arff_data["attributes"].insert(
            -1, (OD_VALUE_NAME, detector.data_type)
        )

        return new_frame

    def select_by_index(self, index: np.array):
        dataframe = self.iloc[index]
        arff_dataframe = DataFrameArff(dataframe.values, columns=self.columns)
        arff_dataframe._arff_data = self._arff_data

        return arff_dataframe

    def select_by_od_quantile(self, quantile):
        value = self[OD_VALUE_NAME].quantile(q=quantile)

        dataframe = self[self[OD_VALUE_NAME] <= value]

        arff_dataframe = DataFrameArff(dataframe.values, columns=self.columns)
        arff_dataframe._arff_data = self._arff_data

        return arff_dataframe


class DataLoader:
    def __init__(self, data_path: str, regex: str = r".*"):
        self._reg = re.compile(regex)
        self.file_paths: List[str] = []
        if os.path.isdir(data_path):
            files = (
                x
                for x in os.listdir(data_path)
                if x.endswith(".arff") and self._reg.match(x)
            )
            for file_name in files:
                self.file_paths.append(os.path.join(data_path, file_name))
        elif (
            os.path.isfile(data_path)
            and data_path.endswith(".arff")
            and self._reg.match(data_path)
        ):
            self.file_paths.append(data_path)

        self.file_paths = sorted(self.file_paths, key=lambda x: os.path.getsize(x))

    @staticmethod
    def _load_data_file(file_path: str):
        raise NotImplementedError()

    @staticmethod
    def _load_arff_file(file_path: str) -> DataFrameArff:
        with open(file_path) as arff_file:
            data = arff.load(arff_file)
            return DataFrameArff(arff_data=data)

    def load_files(self):
        if not self.file_paths:
            raise RuntimeError(
                "No .arff detected. Please specify a correct path and unzip data file."
            )
        for file_path in self.file_paths:
            yield self._load_arff_file(file_path)
