import pandas as pd

def get_labels(dataset):
    rng = pd.date_range('1/1/2014', '1/8/2014', freq='1Min')
    labels = pd.DataFrame(0, index=rng, columns=[1, 2, 3, 4])

    if dataset == 2:
        labels.loc["01-01-2014 0:00":'01-08-2014 0:00', 2:2] = 1
        labels.loc['01-03-2014 0:00':'01-08-2014 0:00', 1:1] = 1
    if dataset == 3:
        labels.loc["01-02-2014 0:00":'01-08-2014 0:00', 2:2] = 1
        labels.loc['01-03-2014 0:00':'01-08-2014 0:00', 3:3] = 1

    return labels