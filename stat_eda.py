import pandas as pd
from sqlalchemy import create_engine
import numpy as np

import matplotlib.pyplot as plt


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
##################             EDA          ##########################
######################################################################

def ecdf(data):
    """Compute ECDF for one-dimensional array of measurments """
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

def permutation_sample(data_1, data_2):
    """ Generate permutation sample from 2 datasets """
    #concatenate 2 datasets
    data_concat = np.concatenate((data_1, data_2))
    #permute data
    permuted_data = np.random.permutation(data_concat)
    #Split the permuted data again
    perm_sample_1 = permuted_data[:len(data_1)]
    perm_sample_2 = permuted_data[len(data_1):]
    return perm_sample_1, perm_sample_2

def draw_perm_reps(data_1, data_2, func, size=1):
    """ Generate multiple êrmutation replicates"""
    #initialize array of replicates
    perm_replicates = np.empty(size)
    for i in range(size):
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)
    return perm_replicates

def diff_of_means(data_1, data_2):
    diff_mean = np.mean(data_1) - np.mean(data_2)
    return diff_mean

def draw_bs_reps(data, func, size=1):
    """ Draw bootstrap replicates """
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = func(np.random.choice(data.dropna(), len(data)))
    return bs_replicates


def draw_bs_pairs_linreg(x, y, size=1):
    #Set up an array of indices
    inds = np.arange(len(x))
    #initialize replicates
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    #generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)
    return bs_slope_reps, bs_intercept_reps


def gen_dash():



    # Fig2
    fig2 = plt.figure(0, figsize=(20,10))
    fig2.add_subplot(321)

    all_data = pd.merge(
            beers,
            breweries,
            left_on=["brewery_id"],
            right_on=["index"],
            sort=True,
            suffixes=["_beer", "_brewery"]
    )

    abv_south, abv_north = all_data.ix[(all_data["region"]=="South")], all_data.ix[(all_data["region"]=="Northeast")]
    abv_south, abv_north = abv_south["abv"], abv_north["abv"]
    x_south, y_south = ecdf(abv_south)
    x_north, y_north = ecdf(abv_north)

    empirical_diff_mean = diff_of_means(abv_south, abv_north)
    perm_reps = draw_perm_reps(abv_south, abv_north, diff_of_means, 10000)
    ########################################################################################
    # !!!!!!!!!!!!!!!!!!      p_value is equal to 0 why???            !!!!!!!!!!!!!!!!!!!!
    ########################################################################################
    p_value = np.sum(perm_reps)
    print(perm_reps)

    _ = plt.plot(x_south, y_south, marker='.', linestyle='none', color='red')
    _ = plt.plot(x_north, y_north, marker='.', linestyle='none', color='blue')
    _ = plt.xlabel('Alcool By Volume')
    _ = plt.ylabel('ECDF')
    _ = plt.text(0.03, 0.8, r'$p_value = %s $' % p_value)
    plt.margins(0.02)

    for i in range(50):
        perm_sample_1, perm_sample_2 = permutation_sample(abv_south, abv_north)
        #compute ECDF
        x_1, y_1 = ecdf(perm_sample_1)
        x_2, y_2 = ecdf(perm_sample_2)
        #plot ECDF
        _ = plt.plot(x_1, y_1, marker='.', linestyle='none', alpha=0.02, color='red')
        _ = plt.plot(x_2, y_2, marker='.', linestyle='none', alpha=0.02, color='blue')


    # Fig 4 visualize variability in linear regression
    # fig4 = plt.figure(0, figsize=(20,10))
    # fig4.add_subplot(322)
    # x = beers["ibu"]
    # y = beers["abv"]
    # beers_free = beers.dropna().reset_index()
    # del beers_free["level_0"]
    # del beers_free["index"]
    # x2 = beers_free["ibu"]
    # y2 = beers_free["abv"]
    # corr_mat = np.corrcoef(x2, y2)
    # a, b = draw_bs_pairs_linreg(beers_free["ibu"], beers_free["abv"], 1000)
    # x_lin = np.array([0, 140])
    #
    # _ = plt.plot(x, y, marker='.', linestyle='none')
    # _ = plt.text(110, 0.12, r'$\rho=%s $' % (round(corr_mat[0,1], 2)))
    # _ = plt.xlabel("IBU")
    # _ = plt.ylabel("Alcool By Volume")
    #
    # for i in range(1000):
    #     _ = plt.plot(x_lin, a[i] * x_lin + b[i], linewidth=0.5, alpha=0.2)


    # Fig 5 draw multiple replicates to see confidence interval & distribution of mean for hypothesis testing
    fig5 = plt.figure(0, figsize=(20, 10))
    fig5.add_subplot(323)

    bs_replicates = draw_bs_reps(beers["abv"], np.mean, 500)
    conf_int = np.percentile(bs_replicates, [2.5, 97.5])
    _ = plt.hist(bs_replicates, normed=True)
    _ = plt.text(0.0600, 1200, r'$conf interval = [%s, %s]$' % (round(conf_int[0], 4), round(conf_int[1], 4)))





    plt.show()



gen_dash()
# print(beers.head(5))
# print(breweries.head(5))
