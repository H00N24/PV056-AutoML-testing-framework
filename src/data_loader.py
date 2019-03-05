import pandas as pd
import os
import re
import numpy as np
from typing import Iterable, List, Dict, Any
import arff
from outlier_detection import DETECTORS
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


class DataFrameArff(pd.DataFrame):
    def __init__(self, arff_data: dict):
        columns = [x[0] for x in arff_data["attributes"]]

        super().__init__(arff_data["data"], columns=columns)

        self._arff_data: Dict[str, Any] = {}
        for key, item in arff_data.items():
            if key.lower() != "data":
                self._arff_data.update({key: item})

    def arff_dumps(self):
        data = self._arff_data
        data.update({"data": self.replace(np.nan, None).values.tolist()})
        return arff.dumps(data)

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
                imputer = SimpleImputer()  # XXX settings
                transformed_data = imputer.fit_transform(
                    self[attr].values.reshape(-1, 1)
                ).reshape(-1, 1)
                columns_index = pd.MultiIndex.from_product(
                    [[attr], [values]], names=["0", "1"]
                )
            else:
                raise ValueError(attr, values)

            new = pd.DataFrame(transformed_data, columns=columns_index)
            if encoded_dataframe.empty:
                encoded_dataframe = new
            else:
                encoded_dataframe = encoded_dataframe.join(new)

        return encoded_dataframe

    def apply_outlier_detectors(self):
        dataframe = self._binarize_categorical_values()
        results = []
        for detector in DETECTORS:
            results.append(
                detector().compute_scores(dataframe, dataframe[dataframe.columns[-1]])
            )

        for detector in results:
            self.insert(
                loc=len(self.columns) - 1, column=detector.name, value=detector.values
            )
            self._arff_data["attributes"].insert(
                -1, (detector.name, detector.data_type)
            )


class DataLoader:
    data_reg = re.compile(r".*\.(arff|data)$")

    def __init__(self, data_paths: Iterable[str]):
        self._data_paths: List[str] = []
        for path in data_paths:
            if os.path.isdir(path):
                files = (
                    x for x in os.listdir(path) if self.data_reg.match(x) is not None
                )
                for file_name in files:
                    self._data_paths.append(path + file_name)
            elif os.path.isfile(path) and self.data_reg.match(path) is not None:
                self._data_paths.append(path)

    @staticmethod
    def _load_data_file(file_path: str):
        raise NotImplementedError()

    @staticmethod
    def _load_arff_file(file_path: str) -> DataFrameArff:
        with open(file_path) as arff_file:
            data = arff.load(arff_file)
            return DataFrameArff(data)

    def load_files(self):
        for file_path in self._data_paths:
            if file_path.endswith(".arff"):
                yield self._load_arff_file(file_path)


if __name__ == "__main__":
    a = DataLoader("../data/")
    dataframe = a._load_arff_file("../data/anneal.arff")
