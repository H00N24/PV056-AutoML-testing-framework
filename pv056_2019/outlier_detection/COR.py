import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri


class CORMetric:

    @staticmethod
    def computeCOR(df):
        pandas2ri.activate()
        r_df = pandas2ri.py2ri(df)
        r.source("../../COR/COR.R")
        cor = robjects.globalenv['CORWrapper']
        r_avgRanks = cor(r_df)
        avgRanks = pandas2ri.ri2py(r_avgRanks)

        return avgRanks.values
