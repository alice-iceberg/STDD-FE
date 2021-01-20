import urllib.request
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

pckg_to_cat_map = {}
cat_list = pd.read_csv('Cat_group.csv')


def in_range(number, start, end):
    if start <= number <= end:
        return True
    else:
        return False


def from_timestamp_to_month(timestamp):
    timestamp = int(timestamp)
    dt = datetime.fromtimestamp(timestamp / 1000)
    month = dt.month
    return month


def from_timestamp_to_day(timestamp):
    timestamp = int(timestamp)
    dt = datetime.fromtimestamp(timestamp / 1000)
    day = dt.day
    return day


def from_timestamp_to_ema_order(timestamp):
    # EMA1 : 22:00:00 - 09:59:59
    # EMA2 : 10:00:00 - 13:59:59
    # EMA3 : 14:00:00 - 17:59:59
    # EMA4 : 18:00:00 - 21:59:59

    timestamp = int(timestamp)
    ema_order = 0

    dt = datetime.fromtimestamp(timestamp / 1000)
    if 0 <= dt.hour < 10 or 22 <= dt.hour <= 23:
        ema_order = 1
    elif 10 <= dt.hour < 14:
        ema_order = 2
    elif 14 <= dt.hour < 18:
        ema_order = 3
    elif 18 <= dt.hour < 22:
        ema_order = 4

    return ema_order


def get_google_category(app_package):
    url = "https://play.google.com/store/apps/details?id=" + app_package
    grouped_category = ""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            source = response.read()

        soup = BeautifulSoup(source, 'html.parser')
        table = soup.find_all("a", {'itemprop': 'genre'})

        genre = table[0].get_text()

        grouped = cat_list[cat_list['App Category'] == genre]['Grouped Category'].values

        if len(grouped) > 0:
            grouped_category = grouped[0]
        else:
            grouped_category = 'NotMapped'
    except Exception:
        grouped_category = 'Unknown or Background'

    finally:
        return grouped_category
