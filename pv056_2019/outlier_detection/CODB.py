import numpy as np
import os
import subprocess
import pandas as pd
import tempfile
import re


class CODBMetric:
    def __init__(self, params):
        self.params = params
        self.weka_jar_path = self.params.pop("jar_path", None)

        if not os.path.exists(self.weka_jar_path):
            raise IOError(
                "Input WEKA-CODB.jar file, '{0}' does not exist.".format(
                    self.weka_jar_path
                )
            )
        self.run_args = ["java", "-jar", self.weka_jar_path]

        for item in self.params.items():
            self.run_args += [item[0], item[1]]

        self.run_args.append("-t")

    def compute_values(self, df: pd.DataFrame, classes: np.array):
        result = np.array([0.0] * df.shape[0])
        fo = tempfile.NamedTemporaryFile()
        df.arff_dump(fo.name)
        output = subprocess.getoutput(
            " ".join(self.run_args + [fo.name, "-n", str(df.shape[0])])
        )
        m = re.findall(r"\d+\. \((\d+)\.\).*COF: ([0-9]*\.?[0-9]*)", output)
        for index, cof in m:
            i = int(index)
            try:
                val = float(cof)
            except ValueError:
                val = np.max(result) + 1
            result[i] = val
        fo.close()
        if result[0] > 0.0:
            result = np.reciprocal(result)

        return result
