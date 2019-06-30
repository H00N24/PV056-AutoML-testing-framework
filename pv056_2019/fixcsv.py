import pandas as pd


def fixdatasets():
    filename = "../datasets.csv"
    df = pd.read_csv(filename, header=None)
    df.to_csv(filename, header=False, index=False)


def fixstatistics():
    filename = "../statistics.csv"
    df = pd.read_csv(filename)
    df.to_csv(filename, header=True, index=False)


if __name__ == '__main__':
    fixdatasets()
    #fixstatistics()
