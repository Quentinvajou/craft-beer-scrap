import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import Subplot
from time import sleep
import numpy as np

# %matplotlib inline

def gen_dashboard():
    # Generate months
    months = []
    for i in range(1,13):
        months.append((i, dt.date(2013, i, 1).strftime('%Y-%m')))

    # Generate data
    t_fixed = np.random.randint(50, size=len(months))
    t_closed = np.random.randint(100, size=len(months))
    t_open = np.random.randint(100, size=len(months))
    t_wip = np.random.randint(50, size=len(months))
    t_wfix = np.random.randint(50, size=len(months))

    # Set x axis
    x = np.array([i[1] for i in months])
    x = np.array([dt.datetime(2013, 9, 28, i, 0) for i in range(12)])
    x = [dt.datetime.strptime(i[1],'%Y-%m').date() for i in months]

    # Create matplotlib figures
    my_dpi = 80

    ### Fig 1 ##########################################################
    fig1 = plt.figure(0,figsize=(2000/my_dpi, 900/my_dpi), dpi=my_dpi)
    fig1.add_subplot(221)
    plt.plot(x, t_fixed, marker='*', linestyle='-', color='g', label='Fixed')
    plt.plot(x, t_closed, marker='*', linestyle='--', color='b', label='Closed')
    plt.fill_between(x,t_closed,0,color='green')
    plt.fill_between(x,t_fixed,0,color='black')

    # Some dates settings
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Label axis
    plt.xlabel('Datum')
    plt.ylabel('# Issues')
    plt.legend(loc="upper right")
    plt.grid(True)

    ### Fig 2 ##########################################################
    fig2 = plt.figure(0, figsize=(2000/my_dpi, 2000/my_dpi), dpi=my_dpi)
    fig2.add_subplot(222)
    plt.plot(x, t_wip, linestyle='--', color='b', label='Work in progress')
    plt.plot(x, t_closed, linestyle='--', color='r', label='Open')
    plt.legend()


    ### Fig 3 ##########################################################
    fig3 = plt.figure(1, figsize=(2000/my_dpi, 1800/my_dpi), dpi=my_dpi)
    fig3.add_subplot(223)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    plt.plot(x, t_wip, linestyle='--', color='b', label='Work in progress')
    plt.plot(x, t_open, linestyle='--', color='r', label='Open')
    plt.fill_between(x,t_wip,0,color='orange')
    #plt.fill_between(x,t_open,0,color='black')
    plt.legend()


    ### Fig 4 #########################################################
    fig4 = plt.figure(1, figsize=(4,4))
    ax = fig4.add_subplot(224)
    #ax = Subplot(fig4, 224
    #ax = plt.axes([0.5, 0.5, 0.4, 0.4])
    # plt.axes([0.5, 0.5, 0.8, 0.8])
    labels = 'Closed', 'Fixed', 'Work in Progress', 'Won\'t Fix', 'Open'
    colors = ('orange', 'green', 'yellow', 'black', 'grey')
    fracs = np.random.randint(50, size=len(labels))
    plt.pie(fracs,labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Issues overview')

    plt.show()


gen_dashboard()
