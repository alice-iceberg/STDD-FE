import os
import urllib.request
from collections import Counter
from datetime import datetime, timedelta
import statistics

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from haversine import haversine, Unit

pckg_to_cat_map = {}
cat_list = pd.read_csv('Cat_group.csv')


def create_filenames(user_id, data_source_ids):
    filenames = {}
    for item, value in enumerate(data_source_ids):
        _filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/data_for_fe/4-{user_id}/{user_id}_{value}.csv'
        filenames[value] = _filename

    return filenames


def add_header_to_df(filename, output_columns):
    df = pd.read_csv(filename)
    df.columns = output_columns
    df.to_csv(filename, index=False)


def reorder_columns_df(filename):
    df = pd.read_csv(filename)
    cols = df.columns.tolist()


def drop_columns_from_df(filename, columns):
    df = pd.read_csv(filename)
    df = df.drop(columns, axis=1)
    df.to_csv(filename, index=False)


def combine_files(directory, output_columns):
    filenames = os.listdir(directory)
    output_dataframe = pd.DataFrame(columns=output_columns)

    for filename in filenames:
        filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/extracted_features/{filename}'
        print("Combining", filename)
        df = pd.read_csv(filename, header=None)
        output_dataframe = output_dataframe.append(df)

    output_dataframe.to_csv('all_extracted_features.csv', index=False)


def in_range(number, start, end):
    if start < number <= end:
        return True
    else:
        return False


def in_range_of_sleep_hours(timestamp):
    # sleep hours are 9pm to 12pm
    # more than 9pm today
    # less than 12 pm tomorrow
    if from_timestamp_to_hour(timestamp) >= 21 or from_timestamp_to_hour(timestamp) <= 12:
        return True
    else:
        return False


def is_next_day(timestamp_now, timestamp_next):
    timestamp_now = int(timestamp_now / 1000)
    timestamp_next = int(timestamp_next / 1000)

    dt_now = datetime.fromtimestamp(timestamp_now)
    dt_next = datetime.fromtimestamp(timestamp_next)

    if dt_next == dt_now + timedelta(days=1):
        return True

    return False


def from_timestamp_to_month(timestamp):
    timestamp = int(timestamp)
    dt = datetime.fromtimestamp(timestamp / 1000)
    month = dt.month
    return month


def from_timestamp_to_year(timestamp):
    timestamp = int(timestamp)
    dt = datetime.fromtimestamp(timestamp / 1000)
    year = dt.year
    return year


def from_timestamp_to_day(timestamp):
    timestamp = int(timestamp)
    dt = datetime.fromtimestamp(timestamp / 1000)
    day = dt.day
    return day


def from_timestamp_to_hour(timestamp):
    timestamp = int(timestamp) / 1000
    dt = datetime.fromtimestamp(int(timestamp))
    hour = dt.hour
    return hour


def is_weekday(timestamp):
    timestamp = int(timestamp) / 1000
    dt = datetime.fromtimestamp(int(timestamp))
    if dt.weekday() == 5 or dt.weekday() == 6:
        return 0
    else:
        return 1


def from_timestamp_to_ema_order(timestamp):
    # EMA1 : 22:00:00 - 09:59:59
    # EMA2 : 10:00:00 - 13:59:59
    # EMA3 : 14:00:00 - 17:59:59
    # EMA4 : 18:00:00 - 21:59:59

    timestamp = int(timestamp)
    ema_order = 0

    dt = datetime.fromtimestamp(timestamp / 1000)
    if dt.hour == 10:
        ema_order = 1
    elif dt.hour == 14:
        ema_order = 2
    elif dt.hour == 18:
        ema_order = 3
    elif dt.hour == 22:
        ema_order = 4

    return ema_order


def get_ema_time_range(ema_timestamp):
    ema_time_range = {
        "time_from": np.NaN,
        "time_to": np.NaN
    }

    ema_order = from_timestamp_to_ema_order(ema_timestamp)
    ema_time_range["time_to"] = ema_timestamp
    if ema_order == 1:
        ema_time_range["time_from"] = ema_timestamp - 3 * 14400000  # 12 hours before EMA
    else:
        ema_time_range["time_from"] = ema_timestamp - 14400000  # 4 hours before EMA

    return ema_time_range


def remove_duplicate_ema(filename):
    ema_arr = []
    dataframe = pd.read_csv(filename, header=None)
    dataframe.columns = ['timestamp', 'value']
    output_dataframe = pd.DataFrame(columns=['timestamp', 'value'])
    for row in dataframe.itertuples():
        date = datetime.fromtimestamp(int(row.timestamp) / 1000).date()
        ema_order = from_timestamp_to_ema_order(row.timestamp)

        if ema_order != 0:
            ema = str(date) + '-' + str(ema_order)
            if ema not in ema_arr:
                ema_arr.append(ema)
                output_dataframe = output_dataframe.append({'timestamp': row.timestamp, 'value': row.value},
                                                           ignore_index=True)

    output_dataframe.to_csv(filename, index=False, header=False)


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


def get_manual_locations(dataframe):
    locations = {
        "home": [],
        "work": [],
        "univ": [],
        "library": [],
        "additional": []
    }

    dataframe = dataframe['value'].str.split(' ', n=3, expand=True)
    dataframe.columns = ['timestamp', 'location', 'lat', 'lng']

    for row in dataframe.itertuples():
        lat = float(row.lat)
        lng = float(row.lng)
        if row.location == "HOME":
            locations["home"] = [lat, lng]
        elif row.location == "WORK":
            locations["work"] = [lat, lng]
        elif row.location == "UNIV":
            locations["univ"] = [lat, lng]
        elif row.location == "LIBRARY":
            locations["library"] = [lat, lng]
        else:
            locations["additional"] = [lat, lng]

    return locations


def get_outlier_cluster(obj, threshold):
    # count number of elements per cluster
    centroids = obj.cluster_centers_
    cnt = Counter(obj.labels_)
    for i, centroid in enumerate(centroids):
        if cnt[i] < threshold:
            return i  # returns an outlier cluster label
    return -1  # returns -1 if no outliers detected


def remove_outlier_cluster(X, outlier_indices):
    deleted_indices = []
    for i, item in enumerate(outlier_indices):
        counter_move_back = 0
        if len(deleted_indices) > 0:
            for deleted_index in deleted_indices:
                if item > deleted_index:
                    counter_move_back += 1
        X = np.delete(X, item - counter_move_back, axis=0)
        deleted_indices.append(item)
    return X


def get_max_distance_from_home(home_location, dataframe, start_time, end_time):
    all_distances_from_home = []

    for row in dataframe.itertuples():
        timestamp = row.timestamp
        if in_range(int(timestamp), start_time, end_time):
            all_distances_from_home.append(
                haversine(home_location, [float(row[1]), float(row[2])], Unit.METERS))

    if len(all_distances_from_home) > 0:
        max_distance_from_home = max(all_distances_from_home)
    else:
        max_distance_from_home = np.nan

    return max_distance_from_home


def get_social_activity_threshold(social_activity_values):
    social_activity_values = np.sort(social_activity_values)
    min_index_separator = round(len(social_activity_values)/3) - 1
    max_index_separator = (len(social_activity_values) - 1) - min_index_separator

    max_social_activity_values = social_activity_values[max_index_separator:]
    social_activity_threshold = round(statistics.mean(max_social_activity_values) / 5)

    return social_activity_threshold



