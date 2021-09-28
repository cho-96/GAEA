import pandas as pd
import geopandas as gpd
from haversine import haversine, Unit
import collections as c
import networkx as nx
import scipy.sparse as scs
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='Data prep')
parser.add_argument('--battery', type=int, default=15, metavar='N', help='Battery range in miles')
args = parser.parse_args()
battery = args.battery

tract = pd.read_csv('data/Washington_demographics.csv')
tract_focus = tract[['Legislative District','White Alone 2010', 'Black or African American Alone 2010', 'Asian Alone 2010']].iloc[1:,:].sort_values(by='Legislative District').reset_index(drop=True)

tract_shp = gpd.read_file('data/Washington_legistrative.shp')
tract_shp_4326 = tract_shp.to_crs(crs=4326)
heliport = pd.read_csv('data/Washington_helipad.xls', delimiter = "\t")
heliport = heliport[heliport['Type']=='HELIPORT']
heliport_loc = heliport[['ARPLatitudeS', 'ARPLongitudeS']]
heliport_loc['lat'] = heliport_loc.apply(lambda x: float(x['ARPLatitudeS'][:-1])/3600 \
                                         if x['ARPLatitudeS'][-1]=='N' \
                                         else -float(x['ARPLatitudeS'][:-1])/3600, axis=1)
heliport_loc['lon'] = heliport_loc.apply(lambda x: float(x['ARPLongitudeS'][:-1])/3600 \
                                         if x['ARPLongitudeS'][-1]=='E' \
                                         else -float(x['ARPLongitudeS'][:-1])/3600, axis=1)
g_heliport_loc = gpd.GeoDataFrame(
    heliport_loc, geometry=gpd.points_from_xy(heliport_loc.lon, heliport_loc.lat))
tract_focus = gpd.GeoDataFrame(
    tract_focus, geometry=tract_shp_4326.geometry)
g_heliport_loc['name'] = range(len(g_heliport_loc))
g_heliport_loc = g_heliport_loc.reset_index(drop=True)
contain = pd.DataFrame(gpd.sjoin(g_heliport_loc, tract_focus)[['name','Legislative District']])
g_heliport_loc = pd.merge(g_heliport_loc, contain, how='left', on='name')

meter_mile = 0.000621371

def edges_from_nodes(dff_nodes):
    sort = c.defaultdict()
    for i, r in dff_nodes.iterrows():
        
        dists = {(r['name'], r1['name']):haversine((r.geometry.x, r.geometry.y), (r1.geometry.x, r1.geometry.y), \
                             unit=Unit.METERS)*meter_mile for j, r1 in dff_nodes.iloc[i+1:].iterrows()}
        dists1 = {(r1['name'], r['name']):haversine((r.geometry.x, r.geometry.y), (r1.geometry.x, r1.geometry.y), \
                             unit=Unit.METERS)*meter_mile for j, r1 in dff_nodes.iloc[i+1:].iterrows()}
        
        sort.update(dists)
        sort.update(dists1)
    sorted_d = sorted(sort.items(), key=lambda kv: kv[1])
    return sorted_d 

sorted_d = edges_from_nodes(g_heliport_loc)
sorted_all = pd.DataFrame(sorted_d)
sorted_all.columns = ['name', 'length']
sorted_below15 = sorted_all[sorted_all['length']<=battery]

def geo_edge_map(G,x,y,weight_dict, weight_fn):
    return weight_fn(weight_dict[(x,y)])

def weight_identity(x):
    return 1

def init_graph(N, adj_dim=2):
    l = [scs.dok_matrix((N, N)) for _ in range(adj_dim)]
    ret = {"adjs" : l, "particles" : c.defaultdict(list), "immunized": set(), "N":N}
    return ret

def networkx_wrapper(G, fn_edges=[lambda G, x,y: npr.exponential(2), lambda G, x,y: np.abs(npr.normal(0,1))], directed=False, aux = {}):


    ret = init_graph(np.max(G.nodes)+1, len(fn_edges))

    for v1, v2 in G.edges:
        for li, f_i in zip(ret["adjs"], fn_edges):
            li[v1, v2] = f_i(G, v1, v2, **aux)
            if not directed:
                li[v2,v1] = li[v1,v2]
            else:
                li[v2,v1] = f_i(G, v2, v1, **aux)
    return ret

def build_geo_graph(df_edges, edge_key="name", weight_key="length", fn_edges=[geo_edge_map, geo_edge_map, geo_edge_map], weight_fn=weight_identity, directed=False):
    edges = df_edges[edge_key]
    weights = df_edges[weight_key]
    weights_dict = {k:v for k,v in zip(edges, weights)}
    return networkx_wrapper(nx.DiGraph(list(edges)), fn_edges=fn_edges, directed=directed, aux={"weight_fn": weight_fn, "weight_dict": weights_dict})

g = build_geo_graph(sorted_below15)

def normalize_graph(g):
    g1 = g.sum(axis=1, keepdims=True)
    g1[np.isnan(g1)] = 1
    g1[g1 == 0] = 1
    return g / g1

prs = [normalize_graph(gg.toarray()) for gg in g["adjs"]]


if __name__ == '__main__':
    dirname = 'data_process'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open('data_process/tract_focus.csv', 'wb') as f:
        tract_focus[['Legislative District', 'White Alone 2010', 'Black or African American Alone 2010', 'Asian Alone 2010']].to_csv(f, index=False)
    with open('data_process/g_heliport_loc.csv', 'wb') as f2:
        g_heliport_loc[['name','Legislative District']].to_csv(f2, index=False)
    with open('below_{}.npy'.format(battery), 'wb') as f3:
        np.save(f3, np.array(prs[0]))
    print(len(sorted_below15))
