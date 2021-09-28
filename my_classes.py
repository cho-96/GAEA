import numpy as np
import pandas as pd
import torch
import collections as c
import numpy.random as npr

tract_focus = pd.read_csv('data_process/tract_focus.csv')
g_heliport_loc = pd.read_csv('data_process/g_heliport_loc.csv')



config = c.defaultdict()
config["particle params"] = [{"df_nodes": g_heliport_loc, "df_tract":tract_focus, "pop_key":"White Alone 2010", "particles":10000},
                                 {"df_nodes": g_heliport_loc, "df_tract":tract_focus, "pop_key":"Black or African American Alone 2010", "particles":10000},
                                 {"df_nodes": g_heliport_loc, "df_tract":tract_focus, "pop_key":"Asian Alone 2010", "particles":10000}]
num_groups = len(config["particle params"])

def init_particle_locs_geo(df_nodes, df_tract, pop_key="white", particles=None):
    helipad_district = np.unique(df_nodes['Legislative District'])     
    df_tract = df_tract[df_tract['Legislative District'].isin(helipad_district)]
    total = np.sum([v for v in df_tract[pop_key]])
    p = [v/total for v in df_tract[pop_key]]
    np.random.seed(2000)
    tids = npr.choice([v['Legislative District'] for i, v in df_tract.iterrows()], particles, p=p, replace=True)
    ret = []
    length_nodes = len(df_nodes)
    one_hot = np.zeros(length_nodes)
    for tid in tids:
        a = [i for i,v in enumerate(df_nodes["Legislative District"]) if v == tid]
        if a:
            np.random.seed(2000)
            one_sample = one_hot
            one_sample[npr.choice(a)] = 1
            ret.append(one_sample)
    return ret

locs = [init_particle_locs_geo(**s) for s in config["particle params"]]

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self):
        'Initialization'
        global locs
        self.particle_locs = locs

    def __len__(self):
        'Denotes the total number of samples'
        return config['particle params'][0]['particles']

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        particles_1 = self.particle_locs[0]
        particles_2 = self.particle_locs[1]
        particles_3 = self.particle_locs[2]

        return particles_1[index], particles_2[index], particles_3[index]


