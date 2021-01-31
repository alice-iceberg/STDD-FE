import os
import statistics
import urllib.request
from collections import Counter
from datetime import datetime, timedelta

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


def combine_files(directory, output_columns):
    if directory != '.DS_Store':
        filenames = os.listdir(directory)
        output_dataframe = pd.DataFrame(columns=output_columns)

        for filename in filenames:
            if filename != '.DS_Store':
                filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/extracted_features2/{filename}'
                print("Combining", filename)
                df = pd.read_csv(filename, header=None, encoding='utf-8')
                output_dataframe = output_dataframe.append(df)

        output_dataframe.to_csv('all_extracted_features.csv', index=False)


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

    ema_order = from_timestamp_to_ema_order(ema_timestamp)
    if ema_order == 1:
        ema_time_range["time_from"].append(ema_timestamp - 3 * 14400000)  # 12 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 28800000)  # 8 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 28800000)  # 8 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp - 14400000)  # 4 hours before EMA

        ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp)
    else:
        ema_time_range["time_from"].append(ema_timestamp - 14400000)  # 4 hours before EMA
        ema_time_range["time_to"].append(ema_timestamp)

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
    df_out['still_freq'] = df.still_freq
    df_out['walking_freq'] = df.walking_freq
    df_out['running_freq'] = df.running_freq
    df_out['on_bicycle_freq'] = df.on_bicycle_freq
    df_out['in_vehicle_freq'] = df.in_vehicle_freq
    df_out['signif_motion_freq'] = df.signif_motion_freq
    df_out['steps_num'] = df.steps_num
    df_out['weekday'] = df.weekday
    df_out['gender'] = df.gender
    df_out['physical_act_gt'] = df.physical_act_gt

    df_out.to_csv('physical_act_features.csv', index=False)


def create_mood_features_file(input_filename):
    df = pd.read_csv(input_filename)
    df_out = pd.DataFrame()
    df_out['user_id'] = df.user_id
    df_out['ema_timestamp'] = df.ema_timestamp
    df_out['still_freq'] = df.still_freq
    df_out['walking_freq'] = df.walking_freq
    df_out['running_freq'] = df.running_freq
    df_out['on_bicycle_freq'] = df.on_bicycle_freq
    df_out['in_vehicle_freq'] = df.in_vehicle_freq
    df_out['signif_motion_freq'] = df.signif_motion_freq
    df_out['steps_num'] = df.steps_num
    df_out['app_entertainment_music_dur'] = df.app_entertainment_music_dur
    df_out['app_utilities_dur'] = df.app_utilities_dur
    df_out['app_shopping_dur'] = df.app_shopping_dur
    df_out['app_games_comics_dur'] = df.app_games_comics_dur
    df_out['app_others_dur'] = df.app_others_dur
    df_out['app_health_wellness_dur'] = df.app_health_wellness_dur
    df_out['app_social_communication_dur'] = df.app_social_communication_dur
    df_out['app_education_dur'] = df.app_education_dur
    df_out['app_travel_dur'] = df.app_travel_dur
    df_out['app_art_design_photo_dur'] = df.app_art_design_photo_dur
    df_out['app_news_magazine_dur'] = df.app_news_magazine_dur
    df_out['app_food_drink_dur'] = df.app_food_drink_dur
    df_out['app_unknown_background_dur'] = df.app_unknown_background_dur
    df_out['app_entertainment_music_freq'] = df.app_entertainment_music_freq
    df_out['app_utilities_freq'] = df.app_utilities_freq
    df_out['app_shopping_freq'] = df.app_shopping_freq
    df_out['app_games_comics_freq'] = df.app_games_comics_freq
    df_out['app_others_freq'] = df.app_others_freq
    df_out['app_health_wellness_freq'] = df.app_health_wellness_freq
    df_out['app_social_communication_freq'] = df.app_social_communication_freq
    df_out['app_education_freq'] = df.app_education_freq
    df_out['app_travel_freq'] = df.app_travel_freq
    df_out['app_art_design_photo_freq'] = df.app_art_design_photo_freq
    df_out['app_news_magazine_freq'] = df.app_news_magazine_freq
    df_out['app_food_drink_freq'] = df.app_food_drink_freq
    df_out['app_unknown_background_freq'] = df.app_unknown_background_freq
    df_out['apps_total_num'] = df.apps_total_num
    df_out['apps_unique_num'] = df.apps_unique_num
    df_out['light_min'] = df.light_min
    df_out['light_max'] = df.light_max
    df_out['light_avg'] = df.light_avg
    df_out['light_stddev'] = df.light_stddev
    df_out['light_dark_ratio'] = df.light_dark_ratio
    df_out['notif_arrived_num'] = df.notif_arrived_num
    df_out['notif_clicked_num'] = df.notif_clicked_num
    df_out['notif_min_dec_time'] = df.notif_min_dec_time
    df_out['notif_max_dec_time'] = df.notif_max_dec_time
    df_out['notif_avg_dec_time'] = df.notif_avg_dec_time
    df_out['notif_stdev_dec_time'] = df.notif_stdev_dec_time
    df_out['screen_on_freq'] = df.screen_on_freq
    df_out['screen_off_freq'] = df.screen_off_freq
    df_out['lock_freq'] = df.lock_freq
    df_out['unlock_freq'] = df.unlock_freq
    df_out['pitch_num'] = df.pitch_num
    df_out['pitch_min'] = df.pitch_min
    df_out['pitch_max'] = df.pitch_max
    df_out['pitch_avg'] = df.pitch_avg
    df_out['pitch_stdev'] = df.pitch_stdev
    df_out['sound_energy_min'] = df.sound_energy_min
    df_out['sound_energy_max'] = df.sound_energy_max
    df_out['sound_energy_avg'] = df.sound_energy_avg
    df_out['sound_energy_stdev'] = df.sound_energy_stdev
    df_out['images_num'] = df.images_num
    df_out['videos_num'] = df.videos_num
    df_out['music_num'] = df.music_num
    df_out['wifi_unique_num'] = df.wifi_unique_num
    df_out['typing_freq'] = df.typing_freq
    df_out['typing_unique_apps_num'] = df.typing_unique_apps_num
    df_out['typing_max'] = df.typing_max
    df_out['typing_avg'] = df.typing_avg
    df_out['typing_stdev'] = df.typing_stdev
    df_out['cal_events_num'] = df.cal_events_num
    df_out['tempC'] = df.tempC
    df_out['totalSnow_cm'] = df.totalSnow_cm
    df_out['cloudcover'] = df.cloudcover
    df_out['precipMM'] = df.precipMM
    df_out['windspeedKmph'] = df.windspeedKmph
    df_out['weekday'] = df.weekday
    df_out['gender'] = df.gender
    df_out['mood_gt'] = df.physical_act_gt

    df_out.to_csv('mood_features.csv', index=False)
