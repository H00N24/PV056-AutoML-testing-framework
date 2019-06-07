import pandas as pd


def fix():
    df = pd.read_csv(filename, header=False)
    df.to_csv(filename, header=False, index=False)


if __name__ == '__main__':
    filename = "datasets.csv"
    fix()
