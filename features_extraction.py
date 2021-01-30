import itertools
import math
import operator
import statistics
from collections import Counter
from datetime import datetime, timedelta
from statistics import mean

import numpy as np
import pandas as pd
from haversine import haversine, Unit
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

import tools


# todo read file once and give dataframe as an input
# todo stop extracting features if ema_order changed


def get_activity_recognition_features(table, start_time, end_time):
    """

    :param table: input dataframe
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of each activity frequency
    """
    activities_features = {
        "still_freq": 0,
        "walking_freq": 0,
        "running_freq": 0,
        "on_bicycle_freq": 0,
        "in_vehicle_freq": 0
    }

    for row in table.itertuples():
        timestamp = int(row.timestamp)

        if tools.in_range(int(timestamp), start_time, end_time):
            if row.value.split(" ")[-1] == 'EXIT':  # todo: solve the problem with only exit values at the beginning
                activity_type = row.value.split(" ")[1]
                if activity_type == 'STILL':
                    activities_features['still_freq'] += 1
                elif activity_type == 'WALKING':
                    activities_features['walking_freq'] += 1
                elif activity_type == 'RUNNING':
                    activities_features['running_freq'] += 1
                elif activity_type == 'ON_BICYCLE':
                    activities_features['on_bicycle_freq'] += 1
                elif activity_type == 'IN_VEHICLE':
                    activities_features['in_vehicle_freq'] += 1

    return activities_features


def get_app_usage_features(table, start_time, end_time):
    """

    :param table: input dataframe
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of duration of each category,
             dict of frequency of each category
             total number of apps
             number of unique apps
    """

    app_usage_features = {
        'app_entertainment_music_dur': 0,
        'app_utilities_dur': 0,
        'app_shopping_dur': 0,
        'app_games_comics_dur': 0,
        'app_others_dur': 0,
        'app_health_wellness_dur': 0,
        'app_social_communication_dur': 0,
        'app_education_dur': 0,
        'app_travel_dur': 0,
        'app_art_design_photo_dur': 0,
        'app_news_magazine_dur': 0,
        'app_food_drink_dur': 0,
        'app_unknown_background_dur': 0,
        'app_entertainment_music_freq': 0,
        'app_utilities_freq': 0,
        'app_shopping_freq': 0,
        'app_games_comics_freq': 0,
        'app_others_freq': 0,
        'app_health_wellness_freq': 0,
        'app_social_communication_freq': 0,
        'app_education_freq': 0,
        'app_travel_freq': 0,
        'app_art_design_photo_freq': 0,
        'app_news_magazine_freq': 0,
        'app_food_drink_freq': 0,
        'app_unknown_background_freq': 0,

        'apps_total_num': 0,
        'apps_unique_num': 0
    }

    apps = []

    for row in table.itertuples():
        start = row.value.split(" ")[0]
        end = row.value.split(" ")[1]
        pckg_name = row.value.split(" ")[2]
        duration = int(end) - int(start)
        if tools.in_range(int(start), start_time, end_time) and duration > 0:
            app_usage_features['apps_total_num'] += 1
            if pckg_name not in apps:
                apps.append(pckg_name)

            if pckg_name in tools.pckg_to_cat_map:
                category = tools.pckg_to_cat_map[pckg_name]
            else:
                category = tools.get_google_category(pckg_name)
                tools.pckg_to_cat_map[pckg_name] = category

            if category == "Entertainment & Music":
                app_usage_features['app_entertainment_music_dur'] += duration
                app_usage_features['app_entertainment_music_freq'] += 1
            elif category == "Utilities":
                app_usage_features['app_utilities_dur'] += duration
                app_usage_features['app_utilities_freq'] += 1
            elif category == "Shopping":
                app_usage_features['app_shopping_dur'] += duration
                app_usage_features['app_shopping_freq'] += 1
            elif category == "Games & Comics":
                app_usage_features['app_games_comics_dur'] += duration
                app_usage_features['app_games_comics_freq'] += 1
            elif category == "Others":
                app_usage_features['app_others_dur'] += duration
                app_usage_features['app_others_freq'] += 1
            elif category == "Health & Wellness":
                app_usage_features['app_health_wellness_dur'] += duration
                app_usage_features['app_health_wellness_freq'] += 1
            elif category == "Social & Communication":
                app_usage_features['app_social_communication_dur'] += duration
                app_usage_features['app_social_communication_freq'] += 1
            elif category == "Education":
                app_usage_features['app_education_dur'] += duration
                app_usage_features['app_education_freq'] += 1
            elif category == "Travel":
                app_usage_features['app_travel_dur'] += duration
                app_usage_features['app_travel_freq'] += 1
            elif category == "Art & Design & Photo":
                app_usage_features['app_art_design_photo_dur'] += duration
                app_usage_features['app_art_design_photo_freq'] += 1
            elif category == "News & Magazine":
                app_usage_features['app_news_magazine_dur'] += duration
                app_usage_features['app_news_magazine_freq'] += 1
            elif category == "Food & Drink":
                app_usage_features['app_food_drink_dur'] += duration
                app_usage_features['app_food_drink_freq'] += 1
            elif category == "Unknown & Background":
                app_usage_features['app_unknown_background_dur'] += duration
                app_usage_features['app_unknown_background_freq'] += 1

    app_usage_features['apps_unique_num'] = len(apps)

    return app_usage_features


def get_light_features(table, start_time, end_time):
    """

        :param table: input dataframe
        :param start_time: start time of needed range
        :param end_time: end time of needed range
        :return: dict light features: min, max, avg, stddev, % of time when light is 0
        """

    light_features = {
        'light_min': np.NaN,
        'light_max': np.NaN,
        'light_avg': np.NaN,
        'light_stddev': np.NaN,
        'light_dark_ratio': np.NaN
    }
    light_data = []

    for row in table.itertuples(index=False):
        timestamp = row.timestamp
        value = row.value.split(" ")[1]
        if tools.in_range(int(timestamp), start_time, end_time):
            light_data.append(float(value))

    if light_data.__len__() > 0:
        light_features['light_min'] = min(light_data)
        light_features['light_max'] = max(light_data)
        light_features['light_avg'] = statistics.mean(light_data)
        light_features['light_dark_ratio'] = light_data.count(0) / len(light_data)

        if light_data.__len__() > 1:
            light_features['light_stddev'] = statistics.stdev(light_data)

    return light_features


def get_signif_motion_features(table, start_time, end_time):
    """

    :param table: input dataframe
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: number of times significant motion sensor is triggered
    """

    signif_motion_freq = 0

    for row in table.itertuples(index=False):
        timestamp = row.timestamp
        if tools.in_range(int(timestamp), start_time, end_time):
            signif_motion_freq += 1

    return signif_motion_freq


def get_step_detector_features(table, start_time, end_time):
    """

    :param table: input dataframe
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: number of steps
    """
    num_of_steps = 0

    for row in table.itertuples(index=False):
        timestamp = row.timestamp
        if tools.in_range(int(timestamp), start_time, end_time):
            num_of_steps += 1

    return num_of_steps


def get_calls_features(table, start_time, end_time):
    """

    :param table: input dataframe
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of call features: "missed_num", "in_num", "out_num", "min_out_dur", "max_out_dur",
    "avg_out_dur" ,"total_out_dur", "min_in_dur", "max_in_dur", "avg_in_dur", "total_in_dur"
    """

    calls_features = {
        'calls_missed_num': 0,
        'calls_in_num': 0,
        'calls_out_num': 0,
        'calls_min_out_dur': 0,
        'calls_max_out_dur': 0,
        'calls_avg_out_dur': 0,
        'calls_total_out_dur': 0,
        'calls_min_in_dur': 0,
        'calls_max_in_dur': 0,
        'calls_avg_in_dur': 0,
        'calls_total_in_dur': 0
    }

    in_calls_dur = []
    out_calls_dur = []

    for row in table.itertuples(index=False):
        timestamp = row.timestamp
        start = row.value.split(" ")[0]
        end = row.value.split(" ")[1]
        call_type = row.value.split(" ")[2]

        if tools.in_range(int(timestamp), start_time, end_time):
            if call_type == "IN":
                in_calls_dur.append(int(end) - int(start))
                calls_features['calls_in_num'] += 1
            elif call_type == "OUT":
                out_calls_dur.append(int(end) - int(start))
                calls_features['calls_out_num'] += 1
            else:
                calls_features['calls_missed_num'] += 1

    if calls_features['calls_out_num'] > 0:
        calls_features['calls_min_out_dur'] = min(out_calls_dur)
        calls_features['calls_max_out_dur'] = max(out_calls_dur)
        calls_features['calls_avg_out_dur'] = statistics.mean(out_calls_dur)
        calls_features['calls_total_out_dur'] = sum(out_calls_dur)

    if calls_features['calls_in_num'] > 0:
        calls_features['calls_min_in_dur'] = min(in_calls_dur)
        calls_features['calls_max_in_dur'] = max(in_calls_dur)
        calls_features['calls_avg_in_dur'] = statistics.mean(in_calls_dur)
        calls_features['calls_total_in_dur'] = sum(in_calls_dur)

    return calls_features


def get_sms_features(table, start_time, end_time):
    """

    :param table: input dataframe
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of sms features: min_chars, max_chars, avg_chars, unique_contacts_num, total_num
    """

    sms_features = {
        'sms_min_chars': 0,
        'sms_max_chars': 0,
        'sms_avg_chars': 0,
        'sms_unique_contacts': 0,
        'sms_total_num': 0
    }

    unique_contacts = []
    chars_arr = []

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            sms_features['sms_total_num'] += 1
            contact = row.value.split(" ")[1]
            chars = int(row.value.split(" ")[2])
            chars_arr.append(chars)

            if contact not in unique_contacts:
                unique_contacts.append(contact)

    if sms_features['sms_total_num'] > 0:
        sms_features['sms_min_chars'] = min(chars_arr)
        sms_features['sms_max_chars'] = max(chars_arr)
        sms_features['sms_avg_chars'] = statistics.mean(chars_arr)
        sms_features['sms_unique_contacts_num'] = len(unique_contacts)

    return sms_features


def get_notifications_features(table, start_time, end_time):
    """

    :param table: input dataframe
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of notification features: arrived_num, clicked_num, min_decision_time, max_decision_time,
    avg_decision_time, stdev_decision_time
    """

    notifications_features = {
        "notif_arrived_num": 0,
        "notif_clicked_num": 0,
        "notif_min_dec_time": np.NaN,
        "notif_max_dec_time": np.NaN,
        "notif_avg_dec_time": np.NaN,
        "notif_stdev_dec_time": np.NaN
    }

    decision_times = []

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            notification_flag = row.value.split(" ")[-1]
            if notification_flag == "ARRIVED":
                notifications_features["notif_arrived_num"] += 1
            elif notification_flag == "DECISION_TIME":
                decision_times.append(int(row.value.split(" ")[1]) - int(row.value.split(" ")[0]))
            elif notification_flag == "CLICKED":
                notifications_features["notif_clicked_num"] += 1

    if len(decision_times) > 0:
        notifications_features["notif_min_dec_time"] = min(decision_times)
        notifications_features["notif_max_dec_time"] = max(decision_times)
        notifications_features["notif_avg_dec_time"] = statistics.mean(decision_times)

        if len(decision_times) > 1:
            notifications_features["notif_stdev_dec_time"] = statistics.stdev(decision_times)

    return notifications_features


def get_screen_state_features(table, start_time, end_time):
    """

    :param table: input table
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of screen features: on_num, off_num
    """

    screen_features = {
        "screen_on_freq": 0,
        "screen_off_freq": 0
    }

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            screen_state_flag = row.value.split(" ")[-1]
            if screen_state_flag == "ON":
                screen_features["screen_on_freq"] += 1
            else:
                screen_features["screen_off_freq"] += 1

    return screen_features


def get_unlock_state_features(table, start_time, end_time):
    """

        :param table: input dataframe
        :param start_time: start time of needed range
        :param end_time: end time of needed range
        :return: dict of unlock state features: lock_num, unlock_num, phone_usage time
        """

    unlock_state_features = {
        "lock_freq": 0,
        "unlock_freq": 0,
    }

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            unlock_state_flag = row.value.split(" ")[-1]
            if unlock_state_flag == "LOCK":
                unlock_state_features["lock_freq"] += 1
            else:
                unlock_state_features["unlock_freq"] += 1

    return unlock_state_features


def get_microphone_features(table, start_time, end_time):
    """

    :param table: input dataframe
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of microphone features: pitch_num, pitch_min, pitch_max, pitch_avg, pitch_stdev,
    energy_min, energy_max, energy_avg, energy_stdev
    """

    microphone_features = {
        "pitch_num": 0,
        "pitch_min": np.NaN,
        "pitch_max": np.NaN,
        "pitch_avg": np.NaN,
        "pitch_stdev": np.NaN,
        "sound_energy_min": np.NaN,
        "sound_energy_max": np.NaN,
        "sound_energy_avg": np.NaN,
        "sound_energy_stdev": np.NaN
    }

    energies = []
    pitches = []

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            microphone_flag = row.value.split(" ")[-1]
            if microphone_flag == "ENERGY":
                energies.append(float(row.value.split(" ")[1]))
            else:
                pitches.append(float(row.value.split(" ")[1]))

    microphone_features["pitch_num"] = len(pitches)
    if microphone_features["pitch_num"] > 0:
        microphone_features["pitch_min"] = min(pitches)
        microphone_features["pitch_max"] = max(pitches)
        microphone_features["pitch_avg"] = statistics.mean(pitches)

        if microphone_features["pitch_num"] > 1:
            microphone_features["pitch_stdev"] = statistics.stdev(pitches)

    if len(energies) > 0:
        microphone_features["sound_energy_min"] = min(energies)
        microphone_features["sound_energy_max"] = max(energies)
        microphone_features["sound_energy_avg"] = statistics.mean(energies)

        if len(energies) > 1:
            microphone_features["sound_energy_stdev"] = statistics.stdev(energies)

    return microphone_features


def get_stored_media_features(table, start_time, end_time):
    """

        :param table: input dataframe
        :param start_time: start time of needed range
        :param end_time: end time of needed range
        :return: dict of stored media features: images_num, videos_num, music_num
        """

    stored_media_features = {
        "images_num": np.NaN,
        "videos_num": np.NaN,
        "music_num": np.NaN
    }

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            media_flag = row.value.split(" ")[-1]
            if media_flag == "IMAGE":
                stored_media_features["images_num"] = row.value.split(" ")[1]
            elif media_flag == "VIDEO":
                stored_media_features["videos_num"] = row.value.split(" ")[1]
            else:
                stored_media_features["music_num"] = row.value.split(" ")[1]

    return stored_media_features


def get_wifi_features(table, start_time, end_time):
    """

    :param table: input dataframe
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: num of unique wifi bssid
    """

    unique_wifi_bssid = []

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            values = row.value.split(" ")[1].replace("[", "").replace("]", "").split(",")
            for value in values:
                if value not in unique_wifi_bssid:
                    unique_wifi_bssid.append(value)

    return len(unique_wifi_bssid)


def get_typing_features(table, start_time, end_time):
    """

    :param table: input dataframe
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of typing features: typing_num, unique_apps_num, typing_min, typing_max, typing_avg, typing_stdev
    """

    typing_features = {
        "typing_freq": 0,
        "typing_unique_apps_num": 0,
        "typing_min": np.NaN,
        "typing_max": np.NaN,
        "typing_avg": np.NaN,
        "typing_stdev": np.NaN
    }

    typing_durations = []
    unique_apps = []

    for row in table.itertuples(index=False):
        timestamp = row.value.split(" ")[0]

        if tools.in_range(int(timestamp), start_time, end_time):
            typing_duration = int(row.value.split(" ")[1]) - int(row.value.split(" ")[0])
            if typing_duration > 0:
                typing_durations.append(typing_duration)
            if row.value.split(" ")[-1] not in unique_apps:
                unique_apps.append(row.value.split(" ")[-1])

    typing_features["typing_freq"] = len(typing_durations)
    typing_features["typing_unique_apps_num"] = len(unique_apps)

    if typing_features["typing_freq"] > 0:
        typing_features["typing_min"] = min(typing_durations)
        typing_features["typing_max"] = max(typing_durations)
        typing_features["typing_avg"] = statistics.mean(typing_durations)

        if typing_features["typing_freq"] > 1:
            typing_features["typing_stdev"] = statistics.stdev(typing_durations)

    return typing_features


def get_calendar_features(table, start_time, end_time):
    # todo: recheck how many calendar records are for each EMA
    num_events = np.NaN

    for row in table.itertuples(index=False):
        timestamp = row.timestamp
        if tools.in_range(int(timestamp), start_time, end_time):
            num_events = row.value.split(" ")[1]

    return num_events


def get_locations_features_old(table, manual_locations_table, start_time, end_time):
    """

    :param manual_locations_table: input dataframe (manual locations)
    :param table: input dataframe (locations gps)
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return:
    """

    location_features = {
        'places_num': np.NaN,
        "dur_at_place_max": np.NaN,
        "dur_at_place_min": np.NaN,
        "dur_at_place_avg": np.NaN,
        "dur_at_place_stdev": np.NaN,
        "entropy": np.NaN,
        "normalized_entropy": np.NaN,
        "location_variance": np.NaN,
        "duration_at_home": np.NaN,
        "duration_at_work/study": np.NaN,
        "distance_btw_locations_max": np.NaN,
        "distance_from_home_max": np.NaN,
        "distance_travelled_total": np.NaN
    }

    MIN_POINTS_PER_CLUSTER = 3  # 3 elements is 15 minutes
    MAX_DISTANCE_IN_CLUSTER = 100  # in meters

    lat_lng = []
    timestamps = []
    max_indices = []
    distances = []
    total_distance_travelled_per_cluster = []
    num_clusters = 1

    table = table['value'].str.split(' ', n=5, expand=True)
    table.columns = ['timestamp', 'lat', 'lng', 'speed', 'accuracy', 'altitude']
    manual_locations = tools.get_manual_locations(manual_locations_table)
    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            lat_lng.append([float(row.lat), float(row.lng)])
            timestamps.append(timestamp)

    lat_lng = np.array(lat_lng).astype('float64')
    timestamps = np.array(timestamps).astype('int64')

    if len(lat_lng) > 3:
        location_features["location_variance"] = lat_lng.var()

        while True:
            temp = num_clusters
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
            y_label = kmeans.fit_predict(lat_lng)

            # squared distance to cluster center
            X_dist = kmeans.transform(lat_lng) ** 2

            for label in np.unique(kmeans.labels_):
                X_label_indices = np.where(y_label == label)[0]
                max_label_idx = X_label_indices[np.argmax(X_dist[y_label == label].sum(axis=1))]
                max_indices.append(max_label_idx)

            farthest_points = (lat_lng[max_indices])
            centroids = kmeans.cluster_centers_

            for i, centroid in enumerate(centroids):
                distances.append(haversine(farthest_points[i], centroid, unit=Unit.METERS))
                # generate new cluster if distance is more than a threshold
                if distances[i] > MAX_DISTANCE_IN_CLUSTER:
                    temp = num_clusters + 1

            total_distance_travelled_per_cluster.append(sum(distances))
            elements_per_cluster = Counter(kmeans.labels_)
            if elements_per_cluster[0] < MIN_POINTS_PER_CLUSTER and temp > 1:
                break
            if temp == num_clusters:
                outlier_label = tools.get_outlier_cluster(kmeans, MIN_POINTS_PER_CLUSTER)
                if outlier_label == -1:  # no outliers detected
                    no_outliers = True
                    location_features['places_num'] = num_clusters
                    break
                else:
                    no_outliers = False
                    indices = np.where(y_label == outlier_label)[0]
                    lat_lng = tools.remove_outlier_cluster(lat_lng, indices)
            elif temp > num_clusters:
                num_clusters = temp
                no_outliers = True  # TODO: check should be true or false
            else:
                print("Exception occurred: temp < num_clusters")
                no_outliers = False

        if no_outliers:

            if manual_locations["home"]:  # exists
                home_cluster_number = kmeans.predict(
                    [manual_locations["home"]])
            if manual_locations["work"]:
                work_cluster_number = kmeans.predict(
                    [manual_locations["work"]])
            if manual_locations["univ"]:
                univ_cluster_number = kmeans.predict(
                    [manual_locations["univ"]])

            # region Time duration per location cluster
            n = num_clusters
            max_timestamp = timestamps[0]
            min_timestamp = timestamps[0]
            time_duration = 0
            time_duration_per_cluster = []  # starts with last cluster
            current_cluster = num_clusters - 1
            elements_per_cluster = Counter(kmeans.labels_)

            while True:
                elems = np.where(y_label == current_cluster)[0]

                for j, value in enumerate(elems):
                    if j == 0:
                        min_timestamp = timestamps[value]
                    elif (j == elements_per_cluster[current_cluster] - 1) and (
                            value - elems[j - 1] > 1):  # if element is one and the last
                        break
                    elif (value - elems[j - 1] > 1) and (elems[j + 1] - value > 1):  # if element is one
                        continue
                    elif j == elements_per_cluster[current_cluster] - 1:  # last element of the cluster reached
                        max_timestamp = timestamps[value]
                        time_duration = time_duration + abs(max_timestamp - min_timestamp)

                    elif j > 0 and abs(elems[j - 1] - value) > 1:
                        max_timestamp = timestamps[elems[j - 1]]

                        if abs(max_timestamp - min_timestamp) != 0:
                            time_duration = time_duration + abs(max_timestamp - min_timestamp)
                            min_timestamp = timestamps[value]
                time_duration_per_cluster.append(time_duration)
                current_cluster = current_cluster - 1  # move to the previous cluster, because started from last

                if current_cluster == -1:  # no clusters remained
                    break

            location_features["dur_at_place_min"] = min(time_duration_per_cluster)
            location_features["dur_at_place_max"] = max(time_duration_per_cluster)
            location_features["dur_at_place_avg"] = statistics.mean(time_duration_per_cluster)

            location_features["distance_travelled_total"] = sum(total_distance_travelled_per_cluster)
            location_features["distance_btw_locations_max"] = max(total_distance_travelled_per_cluster)

            if home_cluster_number == 0:
                location_features["duration_at_home"] = time_duration_per_cluster[-1]  # last index
            elif home_cluster_number > 0:
                home_cluster_index_from_end = -home_cluster_number - 1
                location_features["duration_at_home"] = time_duration_per_cluster[int(
                    home_cluster_index_from_end)]

            if manual_locations["work"] and manual_locations["univ"]:
                location_features["duration_at_work/study"] = 0
                if work_cluster_number == 0:
                    location_features["duration_at_work/study"] += time_duration_per_cluster[-1]  # last index
                elif work_cluster_number > 0:
                    work_cluster_index_from_end = -work_cluster_number - 1
                    location_features["duration_at_work/study"] += time_duration_per_cluster[int(
                        work_cluster_index_from_end)]
                if univ_cluster_number == 0:
                    location_features["duration_at_work/study"] += time_duration_per_cluster[-1]  # last index
                elif univ_cluster_number > 0:
                    univ_cluster_index_from_end = -univ_cluster_number - 1
                    location_features["duration_at_work/study"] += time_duration_per_cluster[int(
                        univ_cluster_index_from_end)]

            elif manual_locations["work"] and not manual_locations["univ"]:
                if work_cluster_number == 0:
                    location_features["duration_at_work/study"] = time_duration_per_cluster[-1]  # last index
                elif work_cluster_number > 0:
                    work_cluster_index_from_end = -work_cluster_number - 1
                    location_features["duration_at_work/study"] = time_duration_per_cluster[int(
                        work_cluster_index_from_end)]

            elif not manual_locations["work"] and manual_locations["univ"]:
                if univ_cluster_number == 0:
                    location_features["duration_at_work/study"] = time_duration_per_cluster[-1]  # last index
                elif univ_cluster_number > 0:
                    univ_cluster_index_from_end = -univ_cluster_number - 1
                    location_features["duration_at_work/study"] = time_duration_per_cluster[int(
                        univ_cluster_index_from_end)]

            # endregion

            # region Location entropy
            percentage_per_cluster = []  # starting from the last
            log_percentage_per_cluster = []
            total_time = sum(time_duration_per_cluster)
            entropy = 0

            for i in range(0, time_duration_per_cluster.__len__()):
                if total_time > 0:
                    percentage_per_cluster.append((time_duration_per_cluster[i] / total_time))
                else:
                    percentage_per_cluster.append(0)
                if percentage_per_cluster[i] != 0:
                    log_percentage_per_cluster.append(np.math.log(percentage_per_cluster[i], 10))  # log10P
                    entropy += percentage_per_cluster[i] * log_percentage_per_cluster[
                        i]
                else:
                    log_percentage_per_cluster.append(0)
                    entropy += 0
                    n -= 1

            location_features["entropy"] = abs(entropy)
            # endregion

            # region normalized entropy
            if n != 0:
                log_num_clusters = math.log(n, 10)
            else:
                log_num_clusters = 0
            if log_num_clusters != 0:
                location_features["normalized_entropy"] = location_features["entropy"] / log_num_clusters
            # endregion

    if manual_locations['home']:
        location_features["distance_from_home_max"] = tools.get_max_distance_from_home(manual_locations["home"], table,
                                                                                       start_time, end_time)

    return location_features


def get_locations_features(table_gps, table_manual_locations, start_time, end_time):
    LOCATION_HOME = "HOME"
    LOCATION_WORK = "WORK"
    LOCATION_LIBRARY = "LIBRARY"
    LOCATION_UNIVERSITY = "UNIV"
    LOCATION_ADDITIONAL = "ADDITIONAL"

    location_features = {
        'num_of_places': np.nan,
        'max_dur_at_place': np.nan,
        'min_dur_at_place': np.nan,
        'avg_dur_at_place': np.nan,
        'stdev_dur_at_place': np.nan,
        'var_dur_at_place': np.nan,
        'dur_of_homestay': np.nan,
        'dur_at_work_study': np.nan,
        'entropy': np.nan,
        'normalized_entropy': np.nan,
        'max_dist_from_home': np.nan,
        'avg_dist_from_home': np.nan,
        'max_dist_btw_places': np.nan,
        'total_dist_travelled': np.nan,
    }

    # region variables and constants
    max_distance_btw_clusters = 0.2
    kms_per_radian = 6371.0088
    eps = max_distance_btw_clusters / kms_per_radian  # radius for dbscan
    min_samples = 4  # min number of samples per cluster for dbscan
    # endregion

    # region location clusters
    df_gps = table_gps['value'].str.split(' ', n=5, expand=True)
    df_gps.columns = ['timestamp', 'lat', 'lng', 'speed', 'accuracy', 'altitude']
    df_gps = df_gps.sort_values(by=['timestamp'])

    X = []
    timestamps = []
    for row in df_gps.itertuples(index=False):
        if tools.in_range(int(row.timestamp), start_time, end_time):
            X.append([row.lat, row.lng])
            timestamps.append(row.timestamp)

    X = np.array(X).astype('float64')
    timestamps = np.array(timestamps).astype('int64')

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine')

    model = dbscan.fit(np.radians(X))
    labels = model.labels_
    unique_labels = set(labels)

    # identifying the points which makes up core points
    sample_cores = np.zeros_like(labels, dtype=bool)
    sample_cores[dbscan.core_sample_indices_] = True
    location_features['num_of_places'] = len(unique_labels) - 1  # minus label '-1'

    cluster_location = {}  # todo optimize (assign cluster value only once)
    for index, label in enumerate(labels):
        cluster_location[label] = X[index]

    # max distance btw two places
    distances_btw_clusters = []
    for loc1, loc2 in itertools.combinations(cluster_location.values(), 2):
        distances_btw_clusters.append(
            haversine([float(loc1[0]), float(loc1[1])], [float(loc2[0]), float(loc2[1])], Unit.METERS))

    location_features['max_dist_btw_places'] = max(distances_btw_clusters)

    # endregion

    # region manual locations
    df_manual = table_manual_locations['value'].str.split(' ', n=3, expand=True)
    df_manual.columns = ['timestamp', 'location', 'lat', 'lng']

    manual_locations = {
        'home': np.nan,
        'work': np.nan,
        'univ': np.nan,
        'library': np.nan,
        'additional': np.nan
    }

    for row in df_manual.itertuples():
        if row.location == LOCATION_HOME:
            manual_locations['home'] = [float(row.lat), float(row.lng)]
        elif row.location == LOCATION_WORK:
            manual_locations['work'] = [float(row.lat), float(row.lng)]
        elif row.location == LOCATION_UNIVERSITY:
            manual_locations['univ'] = [float(row.lat), float(row.lng)]
        elif row.location == LOCATION_LIBRARY:
            manual_locations['library'] = [float(row.lat), float(row.lng)]
        elif row.location == LOCATION_ADDITIONAL:
            manual_locations['additional'] = [float(row.lat), float(row.lng)]

    # endregion

    # region distance from home
    lat_lng_no_outliers = []
    for i, v in enumerate(sample_cores):
        if v:
            lat_lng_no_outliers.append(X[i])

    all_distances_from_home = []
    for lat, lng in lat_lng_no_outliers:
        all_distances_from_home.append(haversine(manual_locations['home'], [float(lat), float(lng)], Unit.METERS))

    location_features['max_dist_from_home'] = max(all_distances_from_home)
    location_features['avg_dist_from_home'] = mean(all_distances_from_home)

    # getting index of the closest point to home
    min_distance_index = all_distances_from_home.index(min(all_distances_from_home))
    closest_lat = lat_lng_no_outliers[min_distance_index][0]
    closest_lng = lat_lng_no_outliers[min_distance_index][1]
    closest_loc_index_X = np.where((X[:, 0] == closest_lat) & (X[:, 1] == closest_lng))
    home_cluster = labels[closest_loc_index_X]
    # endregion

    # region total distance travelled
    all_distances = []
    for index, value in enumerate(lat_lng_no_outliers):
        if index + 1 != len(lat_lng_no_outliers):
            all_distances.append(
                haversine([float(value[0]), float(value[1])], [float(lat_lng_no_outliers[index + 1][0]),
                                                               float(lat_lng_no_outliers[index + 1][1])],
                          Unit.METERS))
    location_features['total_dist_travelled'] = sum(all_distances)
    # endregion

    # region calculating duration at place

    total_time_per_label = {}
    time_per_label = []
    timestamps_per_label = []

    for label in unique_labels:
        if label != -1:
            for i, v in enumerate(labels):
                if v == label:
                    if len(labels) - 1 >= i + 1:  # checking whether the element is the last one
                        if labels[i + 1] == label:
                            timestamps_per_label.append(timestamps[i])
                        else:
                            if len(timestamps_per_label) != 0:
                                timestamps_per_label.append(timestamps[i])
                                time_per_label.append(timestamps_per_label[-1] - timestamps_per_label[0])
                                timestamps_per_label = []

                    else:
                        if len(timestamps_per_label) > 1:
                            time_per_label.append(timestamps_per_label[-1] - timestamps_per_label[0])
                            timestamps_per_label = []

            total_time_per_label[label] = sum(time_per_label)
            time_per_label = []

    location_features['max_dur_at_place'] = total_time_per_label[
        max(total_time_per_label.items(), key=operator.itemgetter(1))[0]]
    location_features['min_dur_at_place'] = total_time_per_label[
        min(total_time_per_label.items(), key=operator.itemgetter(1))[0]]
    location_features['avg_dur_at_place'] = mean(total_time_per_label[k] for k in total_time_per_label)
    location_features['dur_of_homestay'] = total_time_per_label[int(home_cluster)]

    # calculating standard deviation
    durations_list = []
    # appending all the values in the list
    for value in total_time_per_label.values():
        durations_list.append(value)

    location_features['stdev_dur_at_place'] = np.std(durations_list)
    location_features['var_dur_at_place'] = np.var(durations_list)

    # duration at work/ study place
    if not pd.isna(manual_locations['univ']) and pd.isna(manual_locations['work']):  # if only study place
        all_distances_from_univ = []
        for lat, lng in lat_lng_no_outliers:
            all_distances_from_univ.append(haversine(manual_locations['univ'], [float(lat), float(lng)], Unit.METERS))

        # getting index of the closest point to univ
        min_distance_index = all_distances_from_univ.index(min(all_distances_from_univ))
        closest_lat = lat_lng_no_outliers[min_distance_index][0]
        closest_lng = lat_lng_no_outliers[min_distance_index][1]
        closest_loc_index_X = np.where((X[:, 0] == closest_lat) & (X[:, 1] == closest_lng))
        univ_cluster = labels[closest_loc_index_X]

        location_features['dur_at_work_study'] = total_time_per_label[int(univ_cluster)]
    elif not pd.isna(manual_locations['work']) and pd.isna(manual_locations['univ']):  # if only work place
        all_distances_from_work = []
        for lat, lng in lat_lng_no_outliers:
            all_distances_from_work.append(haversine(manual_locations['work'], [float(lat), float(lng)], Unit.METERS))

        # getting index of the closest point to univ
        min_distance_index = all_distances_from_work.index(min(all_distances_from_work))
        closest_lat = lat_lng_no_outliers[min_distance_index][0]
        closest_lng = lat_lng_no_outliers[min_distance_index][1]
        closest_loc_index_X = np.where((X[:, 0] == closest_lat) & (X[:, 1] == closest_lng))
        work_cluster = labels[closest_loc_index_X]

        location_features['dur_at_work_study'] = total_time_per_label[int(work_cluster)]

    if not pd.isna(manual_locations['univ']) and not pd.isna(manual_locations['work']):  # if only both univ and work
        all_distances_from_univ = []
        for lat, lng in lat_lng_no_outliers:
            all_distances_from_univ.append(haversine(manual_locations['univ'], [float(lat), float(lng)], Unit.METERS))

        # getting index of the closest point to univ
        min_distance_index = all_distances_from_univ.index(min(all_distances_from_univ))
        closest_lat = lat_lng_no_outliers[min_distance_index][0]
        closest_lng = lat_lng_no_outliers[min_distance_index][1]
        closest_loc_index_X = np.where((X[:, 0] == closest_lat) & (X[:, 1] == closest_lng))
        univ_cluster = labels[closest_loc_index_X]

        all_distances_from_work = []
        for lat, lng in lat_lng_no_outliers:
            all_distances_from_work.append(haversine(manual_locations['work'], [float(lat), float(lng)], Unit.METERS))

        # getting index of the closest point to univ
        min_distance_index = all_distances_from_work.index(min(all_distances_from_work))
        closest_lat = lat_lng_no_outliers[min_distance_index][0]
        closest_lng = lat_lng_no_outliers[min_distance_index][1]
        closest_loc_index_X = np.where((X[:, 0] == closest_lat) & (X[:, 1] == closest_lng))
        work_cluster = labels[closest_loc_index_X]

        location_features['dur_at_work_study'] = total_time_per_label[int(work_cluster)] + total_time_per_label[
            int(univ_cluster)]

    # endregion

    # region location entropy
    percentage_per_label = {}
    log_percentage_per_label = {}
    durations = total_time_per_label.values()
    total_duration = sum(durations)
    entropy = 0

    for k, v in total_time_per_label.items():
        percentage_per_label[k] = (v / total_duration)
        if percentage_per_label[k] != 0:
            log_percentage_per_label[k] = np.math.log(percentage_per_label[k], 10)  # log10P
            entropy += percentage_per_label[k] * log_percentage_per_label[
                k]
        else:
            log_percentage_per_label[k] = 0
            entropy += 0

    location_features['entropy'] = abs(entropy)
    # endregion

    # region normalized entropy
    log_num_clusters_day = math.log(location_features['num_of_places'], 10)
    if log_num_clusters_day != 0:
        location_features['normalized_entropy'] = location_features['entropy'] / log_num_clusters_day

    # endregion


def get_sleep_duration(table):
    """

    :param table: input dataframe (app usage)
    :return: dict of sleep durations
    """
    # filtering timestamps (leave only 9pm ~ 12pm)
    filtered_table = pd.DataFrame(columns=['timestamp', 'value'])

    for row in table.itertuples(index=False):
        timestamp = int(row.timestamp)
        if tools.in_range_of_sleep_hours(timestamp):
            filtered_table = filtered_table.append({'timestamp': row.timestamp, 'value': row.value}, ignore_index=True)

    timestamps = np.sort(np.array(filtered_table['timestamp']))
    sleep_durations = {}
    night_timestamps = []
    yesterday = datetime.fromtimestamp(int(timestamps[0]) / 1000).date()
    today = datetime.fromtimestamp(int(timestamps[0]) / 1000).date() + timedelta(days=1)

    if tools.from_timestamp_to_hour(timestamps[0]) >= 21:
        yesterday = datetime.fromtimestamp(int(timestamps[0]) / 1000).date()
        today = datetime.fromtimestamp(int(timestamps[0]) / 1000).date() + timedelta(days=1)

    elif tools.from_timestamp_to_hour(timestamps[0]) <= 12:
        today = datetime.fromtimestamp(int(timestamps[0]) / 1000).date()
        yesterday = datetime.fromtimestamp(int(timestamps[0]) / 1000).date() - timedelta(days=1)

    for timestamp in timestamps:
        if (tools.from_timestamp_to_hour(timestamp) >= 21 and datetime.fromtimestamp(
                int(timestamp) / 1000).date() == yesterday) or (
                tools.from_timestamp_to_hour(timestamp) <= 12 and datetime.fromtimestamp(int(timestamp) / 1000).date()
                == today):
            night_timestamps.append(timestamp)
        else:
            yesterday = today
            today = yesterday + timedelta(days=1)
            try:
                differences = np.diff(np.array(night_timestamps))
                max_difference = differences.max()
                max_difference_index = np.argmax(differences)
                start_time = night_timestamps[int(max_difference_index)]
                end_time = night_timestamps[int(max_difference_index) + 1]
                start_time = tools.from_timestamp_to_hour(start_time)
                end_time = tools.from_timestamp_to_hour(end_time)
                if max_difference == 0:
                    sleep_durations[yesterday] = np.nan
                else:
                    sleep_durations[yesterday] = [round(max_difference / 60000), start_time,
                                                  end_time]  # convert to minutes
                night_timestamps = [timestamp]
            except ValueError:
                if len(night_timestamps) == 0:
                    sleep_durations[yesterday] = np.nan
                else:
                    night_timestamps = []

    print(sleep_durations)

    return sleep_durations
