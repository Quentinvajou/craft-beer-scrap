import pandas as pd
from sqlalchemy import create_engine
import numpy as np

import networkx as nx


import seaborn as sns
import matplotlib.pyplot as plt
# from lightning import Lightning

sns.set()
# lgn = Lightning()


######################################################################
##################     QUERY ON DATABASE    ##########################
######################################################################

engine = create_engine('postgresql://quentinvajou:root@localhost:5432/craft_beers')

stmt = "SELECT * FROM beers"
beers = pd.read_sql_query(stmt, con=engine)

stmt = "SELECT * FROM breweries"
breweries = pd.read_sql_query(stmt, con=engine)

# print(beers.head(5))
# print(breweries.head(5))
# print(type(beers["abv"]))

######################################################################
##################     DATA VISUALIZATION   ##########################
######################################################################

def ecdf(arg):
    # Number of data points: n
    n = len(arg)
    # x-data for the ECDF: x
    x = np.sort(arg)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

def str_to_color(arg):
    all_values = []
    for possible_value in set(arg.tolist()):
        all_values.append(possible_value)
    all_values = pd.DataFrame(all_values)
    # all_values.rename(inplace=True, columns={"":"style"})
    all_values["color_col"] = pd.Series(np.round(np.random.random(len(all_values[0])), 3), index=all_values.index)
    print(all_values.head(5))
    return all_values


def gen_dash():

    # Fig 1
    fig1 = plt.figure(0, figsize=(20,10))
    fig1.add_subplot(321)
    _ = plt.hist(beers["abv"].dropna())
    _ = plt.xlabel('Alcool By Volume')
    _ = plt.ylabel('Nb of beers')

    # Fig2
    fig2 = plt.figure(0, figsize=(20,10))
    fig2.add_subplot(322)
    x, y = ecdf(beers["abv"])
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel('Alcool By Volume')
    _ = plt.ylabel('ECDF')

    # Fig 3
    fig3 = plt.figure(0, figsize=(20,10))
    fig3 = plt.subplot2grid((3,2),(1,0), colspan=2)
    all_data = pd.merge(
            beers,
            breweries,
            left_on=["brewery_id"],
            right_on=["index"],
            sort=True,
            suffixes=["_beer", "_brewery"]
    )
    print(all_data.head(5))

    _ = sns.boxplot(x="sub region", y="abv", data=all_data)
    _ = sns.swarmplot(x="sub region", y="abv", data=all_data)
    _ = plt.xlabel('Region')
    _ = plt.ylabel('Alcool By Volume')

    # Fig 4
    fig4 = plt.figure(0, figsize=(20,10))
    fig4.add_subplot(325)
    x = beers["ibu"]
    y = beers["abv"]
    beers_free = beers.dropna()
    x2 = beers_free["ibu"]
    y2 = beers_free["abv"]
    corr_mat = np.corrcoef(x2, y2)
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.text(110, 0.12, r'$\rho=%s $' % (round(corr_mat[0,1], 2)))
    _ = plt.xlabel("IBU")
    _ = plt.ylabel("Alcool By Volume")


    plt.show()


def gen_dash_net():

        ################### DASHBOARD 2 -- NETWORK ########################

        all_data = pd.merge(
                beers,
                breweries,
                left_on=["brewery_id"],
                right_on=["index"],
                sort=True,
                suffixes=["_beer", "_brewery"]
        )
        # print(all_data.head(5))

        fig5 = plt.figure(1, figsize=(20,10))
        fig5.add_subplot(111)
        edges = all_data[["style", "sub region"]]
        edges_tuples = [tuple(x) for x in edges.values]



        colorx = str_to_color(all_data["style"])
        print(colorx)



        G = nx.Graph()
        G.add_nodes_from(all_data["style"])
        G.add_edges_from(edges_tuples)
        nx.draw(G, node_color=colorx)

        plt.show()



gen_dash()
# gen_dash_net()
