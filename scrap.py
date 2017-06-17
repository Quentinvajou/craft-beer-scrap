from urllib.request import urlopen

from bs4 import BeautifulSoup
import pandas as pd
import re
from sqlalchemy import create_engine
import psycopg2

# from lightning import Lightning
# lgn = Lightning()

######################################################################
###############        FUNCTION DEFINITION       #####################
######################################################################

################        determine if a table row is a beer entry        ################
def is_beer_entry(table_row):
    row_cells = table_row.findAll('td')
    beer_id = get_beer_id(row_cells[0].text)
    return (len(row_cells) == 8 and beer_id)

################        return the beer entry numerical identifier from the "Entry" column      ################
def get_beer_id(cell_value):
    r = re.match("^(\d{1,4})\.$", cell_value)
    if r and len(r.groups()) == 1:
        beer_id = r.group(1)
        return int(beer_id)
    else:
        return None

def get_all_beers(html_soup):
    beers = []
    all_rows_in_html_page = html_soup.findAll('tr')
    for table_row in all_rows_in_html_page:
        if is_beer_entry(table_row):
            row_cells = table_row.findAll('td')
            beer_entry = {
                "id": get_beer_id(row_cells[0].text),
                "name": row_cells[1].text,
                "brewery_name": row_cells[2].text,
                "brewery_location": row_cells[3].text,
                "sytle": row_cells[4].text,
                "size": row_cells[5].text,
                "abv": row_cells[6].text,
                "ibu": row_cells[7].text
            }
            beers.append(beer_entry)
    return beers



######################################################################
##################        RUN SCRAPER       ##########################
######################################################################


html = urlopen("http://craftcans.com/db.php?search=all&sort=beerid&ord=desc&view=text")
html_soup = BeautifulSoup(html, 'html.parser')
beers_list = get_all_beers(html_soup)


df = pd.DataFrame(beers_list)
# print(df.head(5))


######################################################################
##################            TIDY UP       ##########################
######################################################################

################        Create breweries table with new set of IDs      ################
breweries = df[["brewery_name", "brewery_location"]]
breweries = breweries.drop_duplicates().reset_index(drop=True)
breweries["id"] = breweries.index
# print(breweries.head(5))

################        Clean up Beers table        ################
beers = pd.merge(
        df,
        breweries,
        left_on=["brewery_name", "brewery_location"],
        right_on=["brewery_name", "brewery_location"],
        sort=True,
        suffixes=("_beer", "_brewery")
)
beers = beers[["abv", "ibu", "id_beer", "name", "size", "sytle", "id_brewery"]]
beers_columns_rename = {
        "id_beer": "id",
        "id_brewery": "brewery_id",
        "sytle": "style"
}
beers.rename(inplace=True, columns=beers_columns_rename)
# print(beers.head(5))

################        Split up location in breweries table        ################
breweries["city"] = breweries["brewery_location"].apply(lambda loc : loc.split(",")[0])
breweries["state"] = breweries["brewery_location"].apply(lambda loc : loc.split(",")[1])
breweries = breweries[["brewery_name", "city", "state"]]
breweries.rename(inplace=True, columns={"brewery_name":"name"})
breweries["state"] = breweries["state"].apply(lambda space : space.strip(' '))

################    extract from excel and add location on breweries    ################
new_cols = pd.read_excel("~/Documents/WorkDev/Python/Scraping/CraftBeer scrap/US_REGIONS.xlsx")
# print(new_cols.head(5))
# print(breweries.head(5))
new_cols.rename(inplace=True, columns={"State Code":"state"})
breweries = pd.merge(
        breweries,
        new_cols,
        how='left',
        on=['state'],
        sort=True
)
breweries.rename(inplace=True, columns={"State Name":"state name", "Region":"region", "Sub-Region":"sub region"})
# print(breweries.head(5))


################        Change String to Numerical values       ################
def string_pct_to_float(arg):
    stripped = str(arg).strip('%')
    try:
        return float(stripped)/100
    except ValueError:
        return None

beers["abv"] = beers["abv"].apply(string_pct_to_float)

def string_to_int(arg):
    try:
        return int(arg)
    except ValueError:
        return None

beers["ibu"] = beers["ibu"].apply(string_to_int)
# print(beers.head(5))

# for possible_value in set(beers["size"].tolist()):
#     print(possible_value)

def extract_ounce(arg):
    match = re.match("(\d{1,2}\.*\d*)", arg)
    if match:
        return float(match.group(0))
    else:
        return None

beers["ounces"] = beers["size"].apply(extract_ounce)
del beers["size"]
# print (beers.head(5))


######################################################################
##################  STORE DATA IN POSTGRES  ##########################
######################################################################
engine = create_engine('postgresql://quentinvajou:root@localhost:5432/craft_beers')
connection = engine.connect()

beers.to_sql('beers', engine, if_exists='replace')
breweries.to_sql('breweries', engine, if_exists="replace")

print(engine.table_names())
