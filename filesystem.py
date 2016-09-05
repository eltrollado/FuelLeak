import pandas as pd
import os.path as os
import domain_specific as domain
import pickle


def read_logfile(absolute_path):
    return pd.read_csv(absolute_path, header=None, delimiter=';', engine='c', parse_dates=True, decimal=',')


def get_pistols_data_frame(path, force=False):
    pickle_path = path + "pistols.pickle"
    if os.exists(pickle_path) and not force:
        return pd.read_pickle(pickle_path)
    else:
        pistols = read_logfile(path + "nozzleMeasures.log")
        pistols = pistols.set_index(pd.DatetimeIndex(pistols[0]))
        pistols[5] = pistols[4] + pistols[5]
        data = pd.pivot_table(pistols, values=5, columns=3, index=0, aggfunc='sum')
        data.to_pickle(pickle_path)
        return data


def get_tank_data_frame(path, force=False):
    pickle_path = path + "tank.pickle"
    if os.exists(pickle_path) and not force:
        return pd.read_pickle(pickle_path)
    else:
        tank = read_logfile(path + "tankMeasures.log")
        tank = tank.set_index(pd.DatetimeIndex(tank[0]))
        data = pd.pivot_table(tank, values=5, columns=3, index=0)
        data.to_pickle(pickle_path)
        return data


def get_refuel_data_frame(path, force=False):
    pickle_path = path + "refuel.pickle"
    if os.exists(pickle_path) and not force:
        return pd.read_pickle(pickle_path)
    else:
        tank = read_logfile(path + "refuel.log")
        tank = tank.set_index(pd.DatetimeIndex(tank[0]))
        data = pd.pivot_table(tank, index=0, values=2, columns=1)
        data.to_pickle(pickle_path)
        return data


def get_adj_refuel_data_frame(path, force=False):
    pickle_path = path + "refuel.pickle"
    if os.exists(pickle_path) and not force:
        return pd.read_pickle(pickle_path)
    else:
        ref = read_logfile(path + "refuel.log")
        ref = ref.pivot(0, 1, 2)
        rng = pd.date_range('1/1/2014', '1/8/2014', freq='1Min')
        df = pd.DataFrame(index=rng)
        df = df.join(ref).fillna(0)

        speed = 1000 / 3.0
        tank = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4], dtype=float)
        lastrow = pd.Series([0, 0, 0, 0], index=[1, 2, 3, 4], dtype=float)

        for r in df.iterrows():
            tank += r[1]
            for i in range(1, 5):
                if tank[i] > 0:
                    if tank[i] > speed:
                        r[1][i] = lastrow[i] + speed
                        tank[i] -= speed
                    else:
                        r[1][i] = lastrow[i] + tank[i]
                        tank[i] = 0
                else:
                    r[1][i] = lastrow[i]

            lastrow = r[1]

        df = df.shift(1)
        df.to_pickle(pickle_path)
        return df


def get_data_set(nr, force=False, frq='15Min'):

    fname = '_Zest{}'.format(nr)
    try:
        with open(fname, 'r') as f:
            return pickle.load(f)
    except:
        with open(fname, 'w') as f:
            path = "C:\Users\Filip\PycharmProjects\FuelLeak\dane\Zestaw {}\\".format(nr)

            dst = {'pistols': get_pistols_data_frame(path, force),
                   'tank': get_tank_data_frame(path,force),
                   'labels': domain.get_labels(nr)}
            if nr == 3:
                dst['refuel'] = get_refuel_data_frame(path, force)
            else:
                dst['refuel'] = get_adj_refuel_data_frame(path,force)

            pickle.dump(dst,f)
            return dst


def extract_container_data(dataset, frq='15Min'):

    fname = '_Zest{}-{}'.format(dataset,frq)

    try:
        with open(fname, 'r') as f:
            return pickle.load(f)

    except:
        with open(fname, 'w') as f:
            ds = get_data_set(dataset)
            tank = ds['tank']
            refill = ds['refuel']
            pistols = ds['pistols']
            labels = ds['labels']

            containers = [1,2,3,4]

            d = {}
            rng = pd.date_range('1/1/2014', '1/8/2014', freq=frq)
            for container in containers:
                df = pd.DataFrame(index=rng)
                df = df.join(tank[container], rsuffix='t')
                df = df.join(refill[container], rsuffix='r',)
                df = df.join(pistols[container], rsuffix='p',)
                df = df.join(labels[container], rsuffix='l')
                df.fillna(0, inplace=True)
                df.columns = [1, 2, 3, 'l']
                d[container] = df

            pickle.dump(d, f)
            return d
