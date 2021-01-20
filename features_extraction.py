import statistics

import numpy as np
import pandas as pd

import tools


# todo read file once and give dataframe as an input
# todo stop extracting features if ema_order changed


def get_activity_recognition_features(filename, start_time, end_time):
    """

    :param filename: input filename
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of each activity frequency
    """
    each_category_frequency = {
        "still": 0,
        "walking": 0,
        "running": 0,
        "on_bicycle": 0,
        "in_vehicle": 0
    }

    table = pd.read_csv(filename)
    table.columns = ["timestamp", "value"]

    for row in table.itertuples():
        timestamp = int(row.timestamp)
        activity_type = row.value.split(" ")[1]

        if tools.in_range(int(timestamp), start_time, end_time):

            if activity_type == 'STILL':
                each_category_frequency['still'] += 1
            elif activity_type == 'WALKING':
                each_category_frequency['walking'] += 1
            elif activity_type == 'RUNNING':
                each_category_frequency['running'] += 1
            elif activity_type == 'ON_BICYCLE':
                each_category_frequency['on_bicycle'] += 1
            elif activity_type == 'IN_VEHICLE':
                each_category_frequency['in_vehicle'] += 1

    if each_category_frequency['still'] == 0:
        each_category_frequency['still'] = 0
    if each_category_frequency['walking'] == 0:
        each_category_frequency['walking'] = 0
    if each_category_frequency['running'] == 0:
        each_category_frequency['running'] = 0
    if each_category_frequency['on_bicycle'] == 0:
        each_category_frequency['on_bicycle'] = 0
    if each_category_frequency['in_vehicle'] == 0:
        each_category_frequency['in_vehicle'] = 0

    return each_category_frequency


def get_app_usage_features(filename, start_time, end_time):
    """

    :param filename: input filename
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of duration of each category,
             dict of frequency of each category
             total number of apps
             number of unique apps
    """

    each_category_duration = {
        "Entertainment & Music": 0,
        "Utilities": 0,
        "Shopping": 0,
        "Games & Comics": 0,
        "Others": 0,
        "Health & Wellness": 0,
        "Social & Communication": 0,
        "Education": 0,
        "Travel": 0,
        "Art & Design & Photo": 0,
        "News & Magazine": 0,
        "Food & Drink": 0,
        "Unknown & Background": 0
    }

    each_category_frequency = {
        "Entertainment & Music": 0,
        "Utilities": 0,
        "Shopping": 0,
        "Games & Comics": 0,
        "Others": 0,
        "Health & Wellness": 0,
        "Social & Communication": 0,
        "Education": 0,
        "Travel": 0,
        "Art & Design & Photo": 0,
        "News & Magazine": 0,
        "Food & Drink": 0,
        "Unknown & Background": 0
    }

    total_apps_counter = 0
    apps = []

    table = pd.read_csv(filename)
    table.columns = ["timestamp", "value"]

    for row in table.itertuples():
        start = row.value.split(" ")[0]
        end = row.value.split(" ")[1]
        pckg_name = row.value.split(" ")[2]
        duration = int(end) - int(start)
        if tools.in_range(int(start), start_time, end_time) and duration > 0:
            total_apps_counter += 1
            if pckg_name not in apps:
                apps.append(pckg_name)

            if pckg_name in tools.pckg_to_cat_map:
                category = tools.pckg_to_cat_map[pckg_name]
            else:
                category = tools.get_google_category(pckg_name)
                tools.pckg_to_cat_map[pckg_name] = category

            if category == "Entertainment & Music":
                each_category_duration['Entertainment & Music'] += duration
                each_category_frequency['Entertainment & Music'] += 1
            elif category == "Utilities":
                each_category_duration['Utilities'] += duration
                each_category_frequency['Utilities'] += 1
            elif category == "Shopping":
                each_category_duration['Shopping'] += duration
                each_category_frequency['Shopping'] += 1
            elif category == "Games & Comics":
                each_category_duration['Games & Comics'] += duration
                each_category_frequency['Games & Comics'] += 1
            elif category == "Others":
                each_category_duration['Others'] += duration
                each_category_frequency['Others'] += 1
            elif category == "Health & Wellness":
                each_category_duration['Health & Wellness'] += duration
                each_category_frequency['Health & Wellness'] += 1
            elif category == "Social & Communication":
                each_category_duration['Social & Communication'] += duration
                each_category_frequency['Social & Communication'] += 1
            elif category == "Education":
                each_category_duration['Education'] += duration
                each_category_frequency['Education'] += 1
            elif category == "Travel":
                each_category_duration['Travel'] += duration
                each_category_frequency['Travel'] += 1
            elif category == "Art & Design & Photo":
                each_category_duration['Art & Design & Photo'] += duration
                each_category_frequency['Art & Design & Photo'] += 1
            elif category == "News & Magazine":
                each_category_duration['News & Magazine'] += duration
                each_category_frequency['News & Magazine'] += 1
            elif category == "Food & Drink":
                each_category_duration['Food & Drink'] += duration
                each_category_frequency['Food & Drink'] += 1
            elif category == "Unknown & Background":
                each_category_duration['Unknown & Background'] += duration
                each_category_frequency['Unknown & Background'] += 1

    unique_apps_counter = len(apps)
    return each_category_duration, each_category_frequency, total_apps_counter, unique_apps_counter


def get_light_features(filename, start_time, end_time):
    """

        :param filename: input filename
        :param start_time: start time of needed range
        :param end_time: end time of needed range
        :return: dict light features: min, max, avg, stddev, % of time when light is 0
        """

    light_features = {
        'min': np.nan,
        'max': np.nan,
        'avg': np.nan,
        'stddev': np.nan,
        'dark_ratio': np.nan
    }
    light_data = []
    table = pd.read_csv(filename)
    table.columns = ["timestamp", "value"]

    for row in table.itertuples(index=False):
        timestamp = row.timestamp
        value = row.value.split(" ")[1]
        if tools.in_range(int(timestamp), start_time, end_time):
            light_data.append(int(value))

    if light_data.__len__() > 0:
        light_features['min'] = min(light_data)
        light_features['max'] = max(light_data)
        light_features['avg'] = statistics.mean(light_data)
        light_features['dark_ratio'] = light_data.count(0) / len(light_data)

        if light_data.__len__() > 1:
            light_features['stddev'] = statistics.stdev(light_data)

    return light_features


def get_signif_motion_features(filename, start_time, end_time):
    """

    :param filename: input filename
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: number of times significant motion sensor is triggered
    """

    signif_motion_freq = 0
    table = pd.read_csv(filename)
    table.columns = ["timestamp", "value"]

    for row in table.itertuples(index=False):
        timestamp = row.timestamp
        if tools.in_range(int(timestamp), start_time, end_time):
            signif_motion_freq += 1

    return signif_motion_freq


def get_step_detector_features(filename, start_time, end_time):
    """

    :param filename: input filename
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: number of steps
    """
    num_of_steps = 0
    table = pd.read_csv(filename)
    table.columns = ["timestamp", "value"]

    for row in table.itertuples(index=False):
        timestamp = row.timestamp
        if tools.in_range(int(timestamp), start_time, end_time):
            num_of_steps += 1

    return num_of_steps


def get_calls_features(filename, start_time, end_time):
    """

    :param filename: input filename
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of call features: "missed_num", "in_num", "out_num", "min_out_dur", "max_out_dur",
    "avg_out_dur" ,"total_out_dur", "min_in_dur", "max_in_dur", "avg_in_dur", "total_in_dur"
    """

    calls_features = {
        "missed_num": 0,
        "in_num": 0,
        "out_num": 0,
        "min_out_dur": 0,
        "max_out_dur": 0,
        "avg_out_dur": 0,
        "total_out_dur": 0,
        "min_in_dur": 0,
        "max_in_dur": 0,
        "avg_in_dur": 0,
        "total_in_dur": 0
    }

    table = pd.read_csv(filename)
    table.columns = ["timestamp", "value"]

    in_calls_dur = []
    out_calls_dur = []

    for row in table.itertuples(index=False):
        timestamp = row.timestamp
        start = row.value.split(" ")[0]
        end = row.value.split(" ")[1]
        call_type = row.row.value.split(" ")[2]

        if tools.in_range(int(timestamp), start_time, end_time):
            if call_type == "IN":
                in_calls_dur.append(int(end) - int(start))
                calls_features["in_num"] += 1
            elif call_type == "OUT":
                out_calls_dur.append(int(end) - int(start))
                calls_features["out_num"] += 1
            else:
                calls_features["missed_num"] += 1

    if calls_features["out_num"] > 0:
        calls_features["min_out_dur"] = min(out_calls_dur)
        calls_features["max_out_dur"] = max(out_calls_dur)
        calls_features["avg_out_dur"] = statistics.mean(out_calls_dur)
        calls_features["total_out_dur"] = sum(out_calls_dur)

    if calls_features["in_num"] > 0:
        calls_features["min_in_dur"] = min(in_calls_dur)
        calls_features["max_in_dur"] = max(in_calls_dur)
        calls_features["avg_in_dur"] = statistics.mean(in_calls_dur)
        calls_features["total_in_dur"] = sum(in_calls_dur)

    return calls_features


def get_sms_features(filename, start_time, end_time):
    """

    :param filename: input filename
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of sms features: min_chars, max_chars, avg_chars, unique_contacts_num, total_num
    """

    sms_features = {
        "min_chars": 0,
        "max_chars": 0,
        "avg_chars": 0,
        "unique_contacts_num": 0,
        "total_num": 0
    }

    unique_contacts = []
    chars_arr = []

    table = pd.read_csv(filename)
    table.columns = ["timestamp", "value"]

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            sms_features["total_num"] += 1
            contact = row.value.split(" ")[1]
            chars = row.value.split(" ")[2]
            chars_arr.append(chars)

            if contact not in unique_contacts:
                unique_contacts.append(contact)

    if sms_features["total_num"] > 0:
        sms_features["min_chars"] = min(chars_arr)
        sms_features["max_chars"] = max(chars_arr)
        sms_features["avg_chars"] = statistics.mean(chars_arr)
        sms_features["unique_contacts_num"] = len(unique_contacts)

    return sms_features


def get_notifications_features(filename, start_time, end_time):
    """

    :param filename: input filename
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of notification features: arrived_num, clicked_num, min_decision_time, max_decision_time,
    avg_decision_time, stdev_decision_time
    """

    notifications_features = {
        "arrived_num": 0,
        "clicked_num": 0,
        "min_decision_time": 0,
        "max_decision_time": 0,
        "avg_decision_time": 0,
        "stdev_decision_time": np.Nan
    }

    table = pd.read_csv(filename)
    table.columns = ["timestamp", "value"]

    decision_times = []

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            notification_flag = row.value.split(" ")[-1]
            if notification_flag == "ARRIVED":
                notifications_features["arrived_num"] += 1
            elif notification_flag == "DECISION_TIME":
                decision_times.append(int(row.value.split(" ")[1]) - int(row.value.split(" ")[0]))
            elif notification_flag == "CLICKED":
                notifications_features["clicked_num"] += 1

    if len(decision_times) > 0:
        notifications_features["min_decision_time"] = min(decision_times)
        notifications_features["max_decision_time"] = max(decision_times)
        notifications_features["avg_decision_time"] = statistics.mean(decision_times)

        if len(decision_times) > 1:
            notifications_features["stdev_decision_time"] = statistics.stdev(decision_times)

    return notifications_features


def get_screen_state_features(filename, start_time, end_time):
    """

    :param filename: input filename
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of screen features: on_num, off_num
    """

    screen_features = {
        "on_num": 0,
        "off_num": 0
    }

    table = pd.read_csv(filename)
    table.columns = ["timestamp", "value"]

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            screen_state_flag = row.value.split(" ")[-1]
            if screen_state_flag == "ON":
                screen_features["on_num"] += 1
            else:
                screen_features["off_num"] += 1

    return screen_features


def get_unlock_state_features(filename, start_time, end_time):
    """

        :param filename: input filename
        :param start_time: start time of needed range
        :param end_time: end time of needed range
        :return: dict of unlock state features: lock_num, unlock_num
        """

    unlock_state_features = {
        "lock_num": 0,
        "unlock_num": 0,
    }

    table = pd.read_csv(filename)
    table.columns = ["timestamp", "value"]

    for row in table.itertuples(index=False):
        timestamp = row.timestamp

        if tools.in_range(int(timestamp), start_time, end_time):
            unlock_state_flag = row.value.split(" ")[-1]
            if unlock_state_flag == "LOCK":
                unlock_state_features["lock_num"] += 1
            else:
                unlock_state_features["unlock_num"] += 1

    return unlock_state_features


def get_microphone_features(filename, start_time, end_time):
    """

    :param filename: input filename
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return: dict of microphone features: pitch_num, pitch_min, pitch_max, pitch_avg, pitch_stdev,
    energy_min, energy_max, energy_avg, energy_stdev
    """

    microphone_features = {
        "pitch_num": 0,
        "pitch_min": np.Nan,
        "pitch_max": np.Nan,
        "pitch_avg": np.Nan,
        "pitch_stdev": np.Nan,
        "energy_min": np.Nan,
        "energy_max": np.Nan,
        "energy_avg": np.Nan,
        "energy_stdev": np.Nan
    }

    energies = []
    pitches = []

    table = pd.read_csv(filename, header=False)
    table.columns = ["timestamp", "value"]

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
        microphone_features["energy_min"] = min(energies)
        microphone_features["energy_max"] = max(energies)
        microphone_features["energy_avg"] = statistics.mean(energies)

        if len(energies) > 1:
            microphone_features["energy_stdev"] = statistics.stdev(energies)

    return microphone_features


def get_keystroke_log_features(filename, start_time, end_time):
    """

    :param filename: input filename
    :param start_time: start time of needed range
    :param end_time: end time of needed range
    :return:
    """

    keystroke_features = {
        "avg_interkey_delay": np.NaN,
        "stdev_interkey_delay": np.NaN,
        "backspace_ratio": np.NaN,
        "autocorrect_num": 0
    }

    table = pd.read_csv(filename, header=False)
    table.columns = ["timestamp", "value"]
