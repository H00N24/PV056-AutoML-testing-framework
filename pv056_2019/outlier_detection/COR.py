import pandas as pd
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri


class CORMetric:

    @staticmethod
    def computeCOR(df):
        path = "tmpdataset.csv"
        df.to_csv(path, index=False)

        r = robjects.r
        r_data = robjects.DataFrame.from_csvfile(path)
        r.source("../COR/COR.R")

        pandas2ri.activate()
        r_result = r.CORWrapper(r_data)
        result = pandas2ri.ri2py(r_result)

        return result
