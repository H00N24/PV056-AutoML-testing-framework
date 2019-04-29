from __future__ import absolute_import

import os
import warnings
from typing import Any, Dict, Iterable, List, Union, Optional

import arff
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from pv056_2019.outlier_detection import DETECTORS
from pv056_2019.utils import ID_NAME

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

                real_values = self[attr].values
                if real_values.dtype == np.dtype("O"):
                    imputer = imputer.set_params(
                        missing_values=None, strategy="constant", fill_value=0
                    )

                transformed_data = imputer.fit_transform(
                    real_values.reshape(-1, 1)
                ).reshape(-1, 1)
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

    def apply_outlier_detectors(
        self, detectors: Dict[str, Dict[str, Union[str, int, float]]]
    ):
        results = []
        for name, settings in detectors.items():
            detector = DETECTORS[name](**settings)
            results.append(detector.compute_scores(self, self[self.columns[-1]]))

        for detector in results:
            self.insert(
                loc=len(self.columns) - 1, column=detector.name, value=detector.values
            )
            self._arff_data["attributes"].insert(
                -1, (detector.name, detector.data_type)
            )

        return self

    def select_by_index(self, index: np.array):
        dataframe = self.iloc[index]
        arff_dataframe = DataFrameArff(dataframe.values, columns=self.columns)
        arff_dataframe._arff_data = self._arff_data

        return arff_dataframe


class DataLoader:
    def __init__(self, data_paths: Iterable[str]):
        self.file_paths: List[str] = []
        for path in data_paths:
            if os.path.isdir(path):
                files = (x for x in os.listdir(path) if x.endswith(".arff"))
                for file_name in files:
                    self.file_paths.append(os.path.join(path, file_name))
            elif os.path.isfile(path) and path.endswith(".arff"):
                self.file_paths.append(path)

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
