import os
import statistics
import urllib.request
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from haversine import haversine, Unit
from wwo_hist import retrieve_hist_data

pckg_to_cat_map = {}
cat_list = pd.read_csv('Cat_group.csv')


def create_filenames(user_id, data_source_ids):
    filenames = {}
    for item, value in enumerate(data_source_ids):
        _filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/59people/4-{user_id}/{user_id}_{value}.csv'
        filenames[value] = _filename

    return filenames


def split_symptom_clusters_file_per_group(filename):
    df = pd.read_csv(filename)
    gb = df.groupby('depr_group')
    df_list = [gb.get_group(x) for x in gb.groups]
    df1 = df_list[0]
    df1 = df1.sort_values(by=['user_id', 'ema_timestamp'])
    df2 = df_list[1]
    df2 = df2.sort_values(by=['user_id', 'ema_timestamp'])
    df3 = df_list[2]
    df3 = df3.sort_values(by=['user_id', 'ema_timestamp'])
    df1.to_csv('symptom_clusters1.csv', index=False)
    df2.to_csv('symptom_clusters2.csv', index=False)
    df3.to_csv('symptom_clusters3.csv', index=False)


def add_header_to_df(filename, output_columns):
    df = pd.read_csv(filename)
    df.columns = output_columns
    df.to_csv(filename, index=False)


def convert_score(score):
    score = int(score)
    return score - 2 * (score - 3)


def reorder_columns_df(filename):
    df = pd.read_csv(filename)
    cols = df.columns.tolist()


def drop_columns_from_df(filename, columns):
    df = pd.read_csv(filename)
    df = df.drop(columns, axis=1)
    df.to_csv(filename, index=False)


def combine_files(directory):
    if directory != '.DS_Store':
        filenames = os.listdir(directory)
        output_dataframe = pd.DataFrame()

        for filename in filenames:
            if filename != '.DS_Store':
                filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/features_until_EMA/{filename}'
                print("Combining", filename)
                df = pd.read_csv(filename, header=None, skiprows=1, encoding='utf-8')
                output_dataframe = output_dataframe.append(df)

        output_dataframe.to_csv('extracted_features_until_ema.csv', index=False)


def in_range(number, start, end):
    if int(start) < int(number) <= int(end):
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


def get_ema_time_range_30min(ema_timestamp):
    ema_time_range = {
        "time_from": [],
        "time_to": []
    }

    ema_order = from_timestamp_to_ema_order(ema_timestamp)
    if ema_order == 1:
        ema_time_range["time_from"].append(ema_timestamp - 3 * 14400000)  # 12 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 41400000)  # 11.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 41400000)  # 11.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 39600000)  # 11 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 39600000)  # 11 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 37800000)  # 10.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 37800000)  # 10.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 36000000)  # 10 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 36000000)  # 10 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 34200000)  # 9.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 34200000)  # 9.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 32400000)  # 9 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 32400000)  # 9 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 30600000)  # 8.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 30600000)  # 8.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 28800000)  # 8 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 28800000)  # 8 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 27000000)  # 7.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 27000000)  # 7.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 25200000)  # 7 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 25200000)  # 7 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 23400000)  # 6.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 23400000)  # 6.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 21400000)  # 6 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 21400000)  # 6 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 19800000)  # 5.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 19800000)  # 5.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 18000000)  # 5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 18000000)  # 5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 16200000)  # 4.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 16200000)  # 4.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 14400000)  # 4 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 12600000)  # 3.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 12600000)  # 3.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 10800000)  # 3 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 10800000)  # 3 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 9000000)  # 2.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 9000000)  # 2.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 7200000)  # 2 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 7200000)  # 2 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 5400000)  # 1.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 5400000)  # 1.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 3600000)  # 1 hour before EMA

        ema_time_range["time_from"].append(ema_timestamp - 3600000)  # 1 hour before EMA
        ema_time_range["time_to"].append(ema_timestamp - 1800000)  # 30 minutes before EMA

        ema_time_range["time_from"].append(ema_timestamp - 1800000)  # 30 minutes before EMA
        ema_time_range["time_to"].append(ema_timestamp)
    else:
        ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 12600000)  # 3.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 12600000)  # 3.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 10800000)  # 3 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 10800000)  # 3 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 9000000)  # 2.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 9000000)  # 2.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 7200000)  # 2 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 7200000)  # 2 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 5400000)  # 1.5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 5400000)  # 1.5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 3600000)  # 1 hour before EMA

        ema_time_range["time_from"].append(ema_timestamp - 3600000)  # 1 hour before EMA
        ema_time_range["time_to"].append(ema_timestamp - 1800000)  # 30 minutes before EMA

        ema_time_range["time_from"].append(ema_timestamp - 1800000)  # 30 minutes before EMA
        ema_time_range["time_to"].append(ema_timestamp)

    return ema_time_range


def get_ema_time_range_1hr(ema_timestamp):
    ema_time_range = {
        "time_from": [],
        "time_to": []
    }

    ema_order = from_timestamp_to_ema_order(ema_timestamp)
    if ema_order == 1:
        ema_time_range["time_from"].append(ema_timestamp - 3 * 14400000)  # 12 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 39600000)  # 11 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 39600000)  # 11 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 36000000)  # 10 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 36000000)  # 10 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 32400000)  # 9 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 32400000)  # 9 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 28800000)  # 8 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 28800000)  # 8 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 25200000)  # 7 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 25200000)  # 7 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 21400000)  # 6 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 21400000)  # 6 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 18000000)  # 5 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 18000000)  # 5 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 14400000)  # 4 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 10800000)  # 3 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 10800000)  # 3 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 7200000)  # 2 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 7200000)  # 2 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 3600000)  # 1 hour before EMA

        ema_time_range["time_from"].append(ema_timestamp - 3600000)  # 1 hour before EMA
        ema_time_range["time_to"].append(ema_timestamp)
    else:
        ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 10800000)  # 3 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 10800000)  # 3 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 7200000)  # 2 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 7200000)  # 2 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 3600000)  # 1 hour before EMA

        ema_time_range["time_from"].append(ema_timestamp - 3600000)  # 1 hour before EMA
        ema_time_range["time_to"].append(ema_timestamp)

    return ema_time_range


def get_ema_time_range_2hrs(ema_timestamp):
    ema_time_range = {
        "time_from": [],
        "time_to": []
    }

    ema_order = from_timestamp_to_ema_order(ema_timestamp)
    if ema_order == 1:
        ema_time_range["time_from"].append(ema_timestamp - 3 * 14400000)  # 12 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 36000000)  # 10 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 36000000)  # 10 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 28800000)  # 8 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 28800000)  # 8 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 21400000)  # 6 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 21400000)  # 6 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 14400000)  # 4 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 7200000)  # 2 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 7200000)  # 2 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp)
    else:
        ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 7200000)  # 2 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 7200000)  # 2 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp)

    return ema_time_range


def get_ema_time_range_4hrs(ema_timestamp):
    ema_time_range = {
        "time_from": [],
        "time_to": []
    }
    ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
    ema_time_range["time_to"].append(ema_timestamp)
    return ema_time_range


def get_ema_time_range_until_ema(ema_timestamp):
    ema_time_range = {
        "time_from": [],
        "time_to": []
    }
    ema_order = from_timestamp_to_ema_order(ema_timestamp)
    if ema_order == 1:
        ema_time_range['time_from'].append(ema_timestamp - 12 * 60 * 60 * 1000)  # 12 hours before EMA
    elif ema_order == 2:
        ema_time_range['time_from'].append(ema_timestamp - 16 * 60 * 60 * 1000)  # 16 hours before EMA
    elif ema_order == 3:
        ema_time_range['time_from'].append(ema_timestamp - 20 * 60 * 60 * 1000)  # 20 hours before EMA
    elif ema_order == 4:
        ema_time_range['time_from'].append(ema_timestamp - 24 * 60 * 60 * 1000)  # 24 hours before EMA

    ema_time_range["time_to"].append(ema_timestamp)
    return ema_time_range


def get_ema_double_time_range_4hrs(ema_timestamp):
    ema_time_range = {
        "prev_time_from": [],
        "prev_time_to": [],
        "time_from": [],
        "time_to": []
    }
    ema_order = from_timestamp_to_ema_order(ema_timestamp)

    # if ema_order == 1:
    ema_time_range["prev_time_from"].append(ema_timestamp - 4 * 14400000)  # 16 hours before EMA
    ema_time_range["prev_time_to"].append(ema_timestamp - 3 * 14400000)  # 12 hours before EMA
    ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
    ema_time_range["time_to"].append(ema_timestamp)
    # else:
    #     ema_time_range["prev_time_from"].append(ema_timestamp - 2 * 14400000)  # 8 hours before EMA
    #     ema_time_range["prev_time_to"].append(ema_timestamp - 14400000)  # 4 hours before EMA
    #     ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
    #     ema_time_range["time_to"].append(ema_timestamp)

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
    min_index_separator = round(len(social_activity_values) / 3) - 1
    max_index_separator = (len(social_activity_values) - 1) - min_index_separator

    max_social_activity_values = social_activity_values[max_index_separator:]
    social_activity_threshold = round(statistics.mean(max_social_activity_values) / 5)

    return social_activity_threshold


def create_physical_act_features_file(input_filename):
    df = pd.read_csv(input_filename)
    df_out = pd.DataFrame()
    df_out['user_id'] = df.user_id
    df_out['ema_timestamp'] = df.ema_timestamp
    df_out['still_freq_4hr'] = df.still_freq_4hr
    df_out['walking_freq_4hr'] = df.walking_freq_4hr
    df_out['other_act_freq_4hr'] = df.other_act_freq_4hr
    df_out['in_vehicle_freq_4hr'] = df.in_vehicle_freq_4hr
    df_out['still_dur_4hr'] = df.still_dur_4hr
    df_out['walking_dur_4hr'] = df.walking_dur_4hr
    df_out['other_act_dur_4hr'] = df.other_act_dur_4hr
    df_out['in_vehicle_dur_4hr'] = df.in_vehicle_dur_4hr
    df_out['signif_motion_freq_4hr'] = df.signif_motion_freq_4hr
    df_out['steps_num_4hr'] = df.steps_num_4hr
    df_out['still_freq_d'] = df.still_freq_d
    df_out['walking_freq_d'] = df.walking_freq_d
    df_out['other_act_freq_d'] = df.other_act_freq_d
    df_out['in_vehicle_freq_d'] = df.in_vehicle_freq_d
    df_out['still_dur_d'] = df.still_dur_d
    df_out['walking_dur_d'] = df.walking_dur_d
    df_out['other_act_dur_d'] = df.other_act_dur_d
    df_out['in_vehicle_dur_d'] = df.in_vehicle_dur_d
    df_out['signif_motion_freq_d'] = df.signif_motion_freq_d
    df_out['steps_num_d'] = df.steps_num_d
    df_out['weekday'] = df.weekday
    df_out['gender'] = df.gender
    df_out['physical_act_gt'] = df.physical_act_gt

    df_out.to_csv('physical_act_features_all.csv', index=False)


def create_mood_features_file(input_filename):
    df = pd.read_csv(input_filename)
    df_out = pd.DataFrame()
    df_out['user_id'] = df.user_id
    df_out['ema_timestamp'] = df.ema_timestamp
    df_out['still_freq_4hr'] = df.still_freq_4hr
    df_out['walking_freq_4hr'] = df.walking_freq_4hr
    df_out['in_vehicle_freq_4hr'] = df.in_vehicle_freq_4hr
    df_out['other_act_freq_4hr'] = df.other_act_freq_4hr
    df_out['still_dur_4hr'] = df.still_dur_4hr
    df_out['walking_dur_4hr'] = df.walking_dur_4hr
    df_out['in_vehicle_dur_4hr'] = df.in_vehicle_dur_4hr
    df_out['other_act_dur_4hr'] = df.other_act_dur_4hr
    df_out['signif_motion_freq_4hr'] = df.signif_motion_freq_4hr
    df_out['steps_num_4hr'] = df.steps_num_4hr
    df_out['app_entertainment_music_dur_4hr'] = df.app_entertainment_music_dur_4hr
    df_out['app_utilities_dur_4hr'] = df.app_utilities_dur_4hr
    df_out['app_shopping_dur_4hr'] = df.app_shopping_dur_4hr
    df_out['app_games_comics_dur_4hr'] = df.app_games_comics_dur_4hr
    df_out['app_health_wellness_dur_4hr'] = df.app_health_wellness_dur_4hr
    df_out['app_social_communication_dur_4hr'] = df.app_social_communication_dur_4hr
    df_out['app_education_dur_4hr'] = df.app_education_dur_4hr
    df_out['app_travel_dur_4hr'] = df.app_travel_dur_4hr
    df_out['app_art_design_photo_dur_4hr'] = df.app_art_design_photo_dur_4hr
    df_out['app_food_drink_dur_4hr'] = df.app_food_drink_dur_4hr
    df_out['other_app_dur_4hr'] = df.other_app_dur_4hr
    df_out['app_entertainment_music_freq_4hr'] = df.app_entertainment_music_freq_4hr
    df_out['app_utilities_freq_4hr'] = df.app_utilities_freq_4hr
    df_out['app_shopping_freq_4hr'] = df.app_shopping_freq_4hr
    df_out['app_games_comics_freq_4hr'] = df.app_games_comics_freq_4hr
    df_out['app_health_wellness_freq_4hr'] = df.app_health_wellness_freq_4hr
    df_out['app_social_communication_freq_4hr'] = df.app_social_communication_freq_4hr
    df_out['app_education_freq_4hr'] = df.app_education_freq_4hr
    df_out['app_travel_freq_4hr'] = df.app_travel_freq_4hr
    df_out['app_art_design_photo_freq_4hr'] = df.app_art_design_photo_freq_4hr
    df_out['app_food_drink_freq_4hr'] = df.app_food_drink_freq_4hr
    df_out['other_app_freq_4hr'] = df.other_app_freq_4hr
    df_out['apps_total_num_4hr'] = df.apps_total_num_4hr
    df_out['apps_unique_num_4hr'] = df.apps_unique_num_4hr
    df_out['browser_dur_4hr'] = df.browser_dur_4hr
    df_out['light_min_4hr'] = df.light_min_4hr
    df_out['light_max_4hr'] = df.light_max_4hr
    df_out['light_avg_4hr'] = df.light_avg_4hr
    df_out['light_stddev_4hr'] = df.light_stddev_4hr
    df_out['light_dark_ratio_4hr'] = df.light_dark_ratio_4hr
    df_out['num_of_places_4hr'] = df.num_of_places_4hr
    df_out['max_dur_at_place_4hr'] = df.max_dur_at_place_4hr
    df_out['min_dur_at_place_4hr'] = df.min_dur_at_place_4hr
    df_out['avg_dur_at_place_4hr'] = df.avg_dur_at_place_4hr
    df_out['stdev_dur_at_place_4hr'] = df.stdev_dur_at_place_4hr
    df_out['var_dur_at_place_4hr'] = df.var_dur_at_place_4hr
    df_out['dur_of_homestay_4hr'] = df.dur_of_homestay_4hr
    df_out['dur_at_work_study_4hr'] = df.dur_at_work_study_4hr
    df_out['entropy_4hr'] = df.entropy_4hr
    df_out['normalized_entropy_4hr'] = df.normalized_entropy_4hr
    df_out['location_variance_4hr'] = df.location_variance_4hr
    df_out['max_dist_from_home_4hr'] = df.max_dist_from_home_4hr
    df_out['avg_dist_from_home_4hr'] = df.avg_dist_from_home_4hr
    df_out['max_dist_btw_places_4hr'] = df.max_dist_btw_places_4hr
    df_out['total_dist_travelled_4hr'] = df.total_dist_travelled_4hr
    df_out['notif_arrived_num_4hr'] = df.notif_arrived_num_4hr
    df_out['notif_clicked_num_4hr'] = df.notif_clicked_num_4hr
    df_out['notif_min_dec_time_4hr'] = df.notif_min_dec_time_4hr
    df_out['notif_max_dec_time_4hr'] = df.notif_max_dec_time_4hr
    df_out['notif_avg_dec_time_4hr'] = df.notif_avg_dec_time_4hr
    df_out['notif_stdev_dec_time_4hr'] = df.notif_stdev_dec_time_4hr
    df_out['screen_on_freq_4hr'] = df.screen_on_freq_4hr
    df_out['screen_off_freq_4hr'] = df.screen_off_freq_4hr
    df_out['lock_freq_4hr'] = df.lock_freq_4hr
    df_out['unlock_freq_4hr'] = df.unlock_freq_4hr
    df_out['unlock_dur_4hr'] = df.unlock_dur_4hr
    df_out['pitch_num_4hr'] = df.pitch_num_4hr
    df_out['pitch_avg_4hr'] = df.pitch_avg_4hr
    df_out['pitch_stdev_4hr'] = df.pitch_stdev_4hr
    df_out['sound_energy_min_4hr'] = df.sound_energy_min_4hr
    df_out['sound_energy_max_4hr'] = df.sound_energy_max_4hr
    df_out['sound_energy_avg_4hr'] = df.sound_energy_avg_4hr
    df_out['sound_energy_stdev_4hr'] = df.sound_energy_stdev_4hr
    df_out['images_dif_4hr'] = df.images_dif_4hr
    df_out['videos_dif_4hr'] = df.videos_dif_4hr
    df_out['music_dif_4hr'] = df.music_dif_4hr
    df_out['wifi_unique_num_4hr'] = df.wifi_unique_num_4hr
    df_out['typing_freq_4hr'] = df.typing_freq_4hr
    df_out['typing_unique_apps_num_4hr'] = df.typing_unique_apps_num_4hr
    df_out['typing_max_4hr'] = df.typing_max_4hr
    df_out['typing_avg_4hr'] = df.typing_avg_4hr
    df_out['typing_stdev_4hr'] = df.typing_stdev_4hr
    df_out['bkspace_ratio_4hr'] = df.bkspace_ratio_4hr
    df_out['autocor_ratio_4hr'] = df.autocor_ratio_4hr
    df_out['intrkey_delay_avg_4hr'] = df.intrkey_delay_avg_4hr
    df_out['intrkey_delay_stdev_4hr'] = df.intrkey_delay_stdev_4hr
    df_out['key_sessions_num_4hr'] = df.key_sessions_num_4hr
    df_out['gr_x_mean_4hr'] = df.gr_x_mean_4hr
    df_out['gr_x_std_4hr'] = df.gr_x_std_4hr
    df_out['gr_y_mean_4hr'] = df.gr_y_mean_4hr
    df_out['gr_y_std_4hr'] = df.gr_y_std_4hr
    df_out['gr_z_mean_4hr'] = df.gr_z_mean_4hr
    df_out['gr_z_std_4hr'] = df.gr_z_std_4hr
    df_out['cal_events_dif_4hr'] = df.cal_events_dif_4hr

    df_out['still_freq_d'] = df.still_freq_d
    df_out['walking_freq_d'] = df.walking_freq_d
    df_out['in_vehicle_freq_d'] = df.in_vehicle_freq_d
    df_out['other_act_freq_d'] = df.other_act_freq_d
    df_out['still_dur_d'] = df.still_dur_d
    df_out['walking_dur_d'] = df.walking_dur_d
    df_out['in_vehicle_dur_d'] = df.in_vehicle_dur_d
    df_out['other_act_dur_d'] = df.other_act_dur_d
    df_out['signif_motion_freq_d'] = df.signif_motion_freq_d
    df_out['steps_num_d'] = df.steps_num_d
    df_out['app_entertainment_music_dur_d'] = df.app_entertainment_music_dur_d
    df_out['app_utilities_dur_d'] = df.app_utilities_dur_d
    df_out['app_shopping_dur_d'] = df.app_shopping_dur_d
    df_out['app_games_comics_dur_d'] = df.app_games_comics_dur_d
    df_out['app_health_wellness_dur_d'] = df.app_health_wellness_dur_d
    df_out['app_social_communication_dur_d'] = df.app_social_communication_dur_d
    df_out['app_education_dur_d'] = df.app_education_dur_d
    df_out['app_travel_dur_d'] = df.app_travel_dur_d
    df_out['app_art_design_photo_dur_d'] = df.app_art_design_photo_dur_d
    df_out['app_food_drink_dur_d'] = df.app_food_drink_dur_d
    df_out['other_app_dur_d'] = df.other_app_dur_d
    df_out['app_entertainment_music_freq_d'] = df.app_entertainment_music_freq_d
    df_out['app_utilities_freq_d'] = df.app_utilities_freq_d
    df_out['app_shopping_freq_d'] = df.app_shopping_freq_d
    df_out['app_games_comics_freq_d'] = df.app_games_comics_freq_d
    df_out['app_health_wellness_freq_d'] = df.app_health_wellness_freq_d
    df_out['app_social_communication_freq_d'] = df.app_social_communication_freq_d
    df_out['app_education_freq_d'] = df.app_education_freq_d
    df_out['app_travel_freq_d'] = df.app_travel_freq_d
    df_out['app_art_design_photo_freq_d'] = df.app_art_design_photo_freq_d
    df_out['app_food_drink_freq_d'] = df.app_food_drink_freq_d
    df_out['other_app_freq_d'] = df.other_app_freq_d
    df_out['apps_total_num_d'] = df.apps_total_num_d
    df_out['apps_unique_num_d'] = df.apps_unique_num_d
    df_out['browser_dur_d'] = df.browser_dur_d
    df_out['light_min_d'] = df.light_min_d
    df_out['light_max_d'] = df.light_max_d
    df_out['light_avg_d'] = df.light_avg_d
    df_out['light_stddev_d'] = df.light_stddev_d
    df_out['light_dark_ratio_d'] = df.light_dark_ratio_d
    df_out['num_of_places_d'] = df.num_of_places_d
    df_out['max_dur_at_place_d'] = df.max_dur_at_place_d
    df_out['min_dur_at_place_d'] = df.min_dur_at_place_d
    df_out['avg_dur_at_place_d'] = df.avg_dur_at_place_d
    df_out['stdev_dur_at_place_d'] = df.stdev_dur_at_place_d
    df_out['var_dur_at_place_d'] = df.var_dur_at_place_d
    df_out['dur_of_homestay_d'] = df.dur_of_homestay_d
    df_out['dur_at_work_study_d'] = df.dur_at_work_study_d
    df_out['entropy_d'] = df.entropy_d
    df_out['normalized_entropy_d'] = df.normalized_entropy_d
    df_out['location_variance_d'] = df.location_variance_d
    df_out['max_dist_from_home_d'] = df.max_dist_from_home_d
    df_out['avg_dist_from_home_d'] = df.avg_dist_from_home_d
    df_out['max_dist_btw_places_d'] = df.max_dist_btw_places_d
    df_out['total_dist_travelled_d'] = df.total_dist_travelled_d
    df_out['notif_arrived_num_d'] = df.notif_arrived_num_d
    df_out['notif_clicked_num_d'] = df.notif_clicked_num_d
    df_out['notif_min_dec_time_d'] = df.notif_min_dec_time_d
    df_out['notif_max_dec_time_d'] = df.notif_max_dec_time_d
    df_out['notif_avg_dec_time_d'] = df.notif_avg_dec_time_d
    df_out['notif_stdev_dec_time_d'] = df.notif_stdev_dec_time_d
    df_out['screen_on_freq_d'] = df.screen_on_freq_d
    df_out['screen_off_freq_d'] = df.screen_off_freq_d
    df_out['lock_freq_d'] = df.lock_freq_d
    df_out['unlock_freq_d'] = df.unlock_freq_d
    df_out['unlock_dur_d'] = df.unlock_dur_d
    df_out['pitch_num_d'] = df.pitch_num_d
    df_out['pitch_avg_d'] = df.pitch_avg_d
    df_out['pitch_stdev_d'] = df.pitch_stdev_d
    df_out['sound_energy_min_d'] = df.sound_energy_min_d
    df_out['sound_energy_max_d'] = df.sound_energy_max_d
    df_out['sound_energy_avg_d'] = df.sound_energy_avg_d
    df_out['sound_energy_stdev_d'] = df.sound_energy_stdev_d
    df_out['images_dif_d'] = df.images_dif_d
    df_out['videos_dif_d'] = df.videos_dif_d
    df_out['music_dif_d'] = df.music_dif_d
    df_out['wifi_unique_num_d'] = df.wifi_unique_num_d
    df_out['typing_freq_d'] = df.typing_freq_d
    df_out['typing_unique_apps_num_d'] = df.typing_unique_apps_num_d
    df_out['typing_max_d'] = df.typing_max_d
    df_out['typing_avg_d'] = df.typing_avg_d
    df_out['typing_stdev_d'] = df.typing_stdev_d
    df_out['bkspace_ratio_d'] = df.bkspace_ratio_d
    df_out['autocor_ratio_d'] = df.autocor_ratio_d
    df_out['intrkey_delay_avg_d'] = df.intrkey_delay_avg_d
    df_out['intrkey_delay_stdev_d'] = df.intrkey_delay_stdev_d
    df_out['key_sessions_num_d'] = df.key_sessions_num_d
    df_out['gr_x_mean_d'] = df.gr_x_mean_d
    df_out['gr_x_std_d'] = df.gr_x_std_d
    df_out['gr_y_mean_d'] = df.gr_y_mean_d
    df_out['gr_y_std_d'] = df.gr_y_std_d
    df_out['gr_z_mean_d'] = df.gr_z_mean_d
    df_out['gr_z_std_d'] = df.gr_z_std_d
    df_out['cal_events_dif_d'] = df.cal_events_dif_d
    df_out['tempC'] = df.tempC
    df_out['totalSnow_cm'] = df.totalSnow_cm
    df_out['cloudcover'] = df.cloudcover
    df_out['precipMM'] = df.precipMM
    df_out['windspeedKmph'] = df.windspeedKmph
    df_out['weekday'] = df.weekday
    df_out['gender'] = df.gender
    df_out['mood_gt'] = df.mood_gt

    df_out.to_csv('mood_features_all.csv', index=False)

from main import data_directory
def combine_sms_files(user_directory):
    if user_directory != '.DS_Store':
        print(f'Started for {user_directory}')
        user_id = int(user_directory.split('-')[-1])
        sms_file = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/{data_directory}/4-{user_id}/{user_id}_17.csv'
        sms_from_notif_file = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/{data_directory}/4-{user_id}/{user_id}_71.csv'

        if os.stat(sms_from_notif_file).st_size == 0:
            return f'Combining files finished for {user_directory}'

        sms_from_notif_df = pd.read_csv(sms_from_notif_file, header=None)

        if os.stat(sms_file).st_size == 0:
            sms_df = sms_from_notif_df
            sms_df.to_csv(sms_file, index=False)
            return f'Combining files finished for {user_directory}'

        sms_df = pd.read_csv(sms_file)

        sms_df = sms_df.sort_values(by=['timestamp'])
        sms_from_notif_df = sms_from_notif_df.sort_values(by=['timestamp'])

        sms_df = sms_df.drop_duplicates()
        sms_from_notif_df = sms_from_notif_df.drop_duplicates()

        first_timestamp = int(sms_from_notif_df['timestamp'].iloc[0])  # taking the first row
        sms_df['timestamp'] = pd.to_numeric(sms_df['timestamp'])
        sms_df_filtered = sms_df.query(f'timestamp<{first_timestamp}')
        frames = [sms_df_filtered, sms_from_notif_df]

        combined_df = pd.concat(frames)
        combined_df = combined_df.sort_values(by=['timestamp'])

        combined_df.to_csv(sms_file, index=False)

    return f'Combining files finished for {user_directory}'


def remove_mfcc_data(directory):
    from main import data_directory

    user_id = directory.split('-')[1]
    filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/{data_directory}/{directory}/{user_id}_16.csv'
    new_filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/{data_directory}/{directory}/{user_id}_16_new.csv'

    with open(new_filename, 'a+') as write_to:
        with open(filename, 'r') as file:
            for line in file:
                if line[-2] != '"':
                    write_to.write(line)


def get_weather_file(end_date):
    from wwo_hist import retrieve_hist_data
    FREQUENCY = 3
    START_DATE = '01-OCT-2020'
    END_DATE = end_date
    API_KEY = '002b19b9723242eb8e781347211402'
    LOCATION_LIST = ['incheon']

    hist_weather_data = retrieve_hist_data(API_KEY,
                                           LOCATION_LIST,
                                           START_DATE,
                                           END_DATE,
                                           FREQUENCY,
                                           location_label=False,
                                           export_csv=True,
                                           store_df=True)
