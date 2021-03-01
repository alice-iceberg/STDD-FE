import math
import statistics
import time
from datetime import datetime
import concurrent.futures
import os
import numpy as np
import pandas as pd

import features_extraction
import tools

data_directory = '54people'
data_sources = [1, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 71]

user_ids = [
    85,
    86,
    87,
    89,
    90,
    91,
    92,
    93,
    97,
    98,
    99,
    100,
    102,
    103,
    104,
    105,
    107,
    108,
    109,
    110,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    125,
    126,
    128,
    129,
    131,
    132,
    133,
    134,
    135,
    137,
    141,
    143,
    146,
    149,
    150,
    151,
    152,
    154,
    155,
    156,
    159,
    162,
    163,
    164,
    165,
    169,
    170,
    171,
    173
]

user_ids_with_gender = {
    85: 1,
    86: 0,
    87: 1,
    89: 0,
    90: 1,
    91: 1,
    92: 1,
    93: 0,
    97: 0,
    98: 0,
    99: 0,
    100: 1,
    102: 0,
    103: 0,
    104: 0,
    105: 0,
    107: 0,
    108: 1,
    109: 0,
    110: 0,
    113: 0,
    114: 0,
    115: 1,
    116: 0,
    117: 1,
    118: 0,
    119: 1,
    120: 1,
    121: 1,
    122: 0,
    125: 0,
    126: 0,
    128: 1,
    129: 1,
    131: 0,
    132: 0,
    133: 0,
    134: 0,
    135: 0,
    137: 1,
    141: 0,
    143: 0,
    146: 0,
    149: 0,
    150: 1,
    151: 1,
    152: 1,
    154: 0,
    155: 0,
    156: 1,
    159: 1,
    162: 0,
    163: 0,
    164: 0,
    165: 0,
    169: 0,
    170: 1,
    171: 1,
    173: 1
}
user_ids_with_depression_group = {
    85: 2,
    86: 2,
    87: 1,
    89: 1,
    90: 3,
    91: 1,
    92: 2,
    93: 2,
    97: 2,
    98: 1,
    99: 2,
    100: 2,
    102: 1,
    103: 3,
    104: 1,
    105: 2,
    107: 2,
    108: 2,
    109: 2,
    110: 3,
    113: 2,
    114: 1,
    115: 3,
    116: 2,
    117: 1,
    118: 2,
    119: 3,
    120: 1,
    121: 1,
    122: 3,
    125: 3,
    126: 3,
    128: 2,
    129: 1,
    131: 2,
    132: 2,
    133: 1,
    134: 3,
    135: 2,
    137: 3,
    141: 1,
    143: 2,
    146: 2,
    149: 3,
    150: 3,
    151: 3,
    152: 3,
    154: 1,
    155: 2,
    156: 1,
    159: 2,
    162: 2,
    163: 1,
    164: 1,
    165: 3,
    169: 3,
    170: 3,
    171: 1,
    173: 2
}

output_columns = [
    'user_id',
    'ema_timestamp',
    'phq1',
    'phq2',
    'phq3',
    'phq4',
    'phq5',
    'phq6',
    'phq7',
    'phq8',
    'phq9',
    'still_freq',
    'walking_freq',
    'running_freq',
    'on_bicycle_freq',
    'in_vehicle_freq',
    'still_dur',
    'walking_dur',
    'running_dur',
    'on_bicycle_dur',
    'in_vehicle_dur',
    'app_entertainment_music_dur',
    'app_utilities_dur',
    'app_shopping_dur',
    'app_games_comics_dur',
    'app_others_dur',
    'app_health_wellness_dur',
    'app_social_communication_dur',
    'app_education_dur',
    'app_travel_dur',
    'app_art_design_photo_dur',
    'app_news_magazine_dur',
    'app_food_drink_dur',
    'app_unknown_background_dur',
    'app_entertainment_music_freq',
    'app_utilities_freq',
    'app_shopping_freq',
    'app_games_comics_freq',
    'app_others_freq',
    'app_health_wellness_freq',
    'app_social_communication_freq',
    'app_education_freq',
    'app_travel_freq',
    'app_art_design_photo_freq',
    'app_news_magazine_freq',
    'app_food_drink_freq',
    'app_unknown_background_freq',
    'apps_total_num',
    'apps_unique_num',
    'browser_dur',
    'light_min',
    'light_max',
    'light_avg',
    'light_stddev',
    'light_dark_ratio',
    'num_of_places',
    'max_dur_at_place',
    'min_dur_at_place',
    'avg_dur_at_place',
    'stdev_dur_at_place',
    'var_dur_at_place',
    'dur_of_homestay',
    'dur_at_work_study',
    'entropy',
    'normalized_entropy',
    'location_variance',
    'max_dist_from_home',
    'avg_dist_from_home',
    'max_dist_btw_places',
    'total_dist_travelled',
    'signif_motion_freq',
    'steps_num',
    'calls_missed_num',
    'calls_in_num',
    'calls_out_num',
    'calls_min_out_dur',
    'calls_max_out_dur',
    'calls_avg_out_dur',
    'calls_total_out_dur',
    'calls_min_in_dur',
    'calls_max_in_dur',
    'calls_avg_in_dur',
    'calls_total_in_dur',
    'sms_min_chars',
    'sms_max_chars',
    'sms_avg_chars',
    'sms_unique_contacts',
    'sms_total_num',
    'notif_arrived_num',
    'notif_clicked_num',
    'notif_min_dec_time',
    'notif_max_dec_time',
    'notif_avg_dec_time',
    'notif_stdev_dec_time',
    'screen_on_freq',
    'screen_off_freq',
    'lock_freq',
    'unlock_freq',
    'unlock_dur',
    'pitch_num',
    'pitch_avg',
    'pitch_stdev',
    'sound_energy_min',
    'sound_energy_max',
    'sound_energy_avg',
    'sound_energy_stdev',
    'wifi_unique_num',
    'typing_freq',
    'typing_unique_apps_num',
    'typing_min',
    'typing_max',
    'typing_avg',
    'typing_stdev',
    'bkspace_ratio',
    'autocor_ratio',
    'intrkey_delay_avg',
    'intrkey_delay_stdev',
    'key_sessions_num',
    'gr_x_mean',
    'gr_x_std',
    'gr_y_mean',
    'gr_y_std',
    'gr_z_mean',
    'gr_z_std',
    'gender',
    'weekday',  # 1 if weekday, 0 otherwise
    'depr_group',
    'sleep_dur',
    'sleep_start',
    'sleep_end',
    'tempC',
    'totalSnow_cm',
    'cloudcover',
    'precipMM',
    'windspeedKmph',
]
data_sources_with_ids = {
    'ACTIVITY_RECOGNITION': 1,
    'ANDROID_LIGHT': 6,
    'ANDROID_SIGNIFICANT_MOTION': 8,
    'ANDROID_STEP_DETECTOR': 9,
    'ANDROID_WIFI': 27,
    'APPLICATION_USAGE': 28,
    'CALENDAR': 26,
    'CALLS': 29,
    'GRAVITY': 5,
    'INSTAGRAM_FEATURES': 24,
    'KEYSTROKE_LOG': 23,
    'LOCATIONS_MANUAL': 22,
    'LOCATION_GPS': 10,
    'MUSIC_PLAYING': 4,
    'NETWORK_USAGE': 21,
    'NOTIFICATIONS': 20,
    'SCREEN_STATE': 18,
    'SMS': 17,
    'SMS_FROM_NOTIFICATION': 71,
    'SOUND_DATA': 16,
    'STORED_MEDIA': 15,
    'SURVEY_EMA': 11,
    'TYPING': 14,
    'UNLOCK_STATE': 13
}


def extract_features(user_directory):
    if user_directory != '.DS_Store':
        user_id = int(user_directory.split('-')[-1])
        filenames = tools.create_filenames(user_id, data_sources)
        output_table = pd.DataFrame(columns=output_columns)
        output_filename = f'features_until_EMA/extracted_features_{user_id}.csv'
        print(f'Feature extraction started for {user_directory}')

        ema_filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/5people/{user_directory}/{user_id}_11.csv'
        tools.remove_duplicate_ema(ema_filename)

        ema_table = pd.read_csv(ema_filename, delimiter=',', names=['timestamp', 'value'], header=None)
        ema_table = ema_table['value'].str.split(' ', n=10, expand=True)
        ema_table.columns = ['timestamp', 'ema_order', 'phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8',
                             'phq9']
        # region creating dataframes
        activities_dataframe = pd.read_csv(filenames[data_sources_with_ids['ACTIVITY_RECOGNITION']], low_memory=False,
                                           header=None)
        activities_dataframe.columns = ["timestamp", "value"]
        activities_dataframe = activities_dataframe.drop_duplicates()
        activities_dataframe = activities_dataframe.sort_values(by='timestamp')
        app_usage_dataframe = pd.read_csv(filenames[data_sources_with_ids['APPLICATION_USAGE']], low_memory=False,
                                          header=None)
        app_usage_dataframe.columns = ["timestamp", "value"]
        app_usage_dataframe = app_usage_dataframe.drop_duplicates()
        app_usage_dataframe = app_usage_dataframe.sort_values(by='timestamp')
        light_dataframe = pd.read_csv(filenames[data_sources_with_ids['ANDROID_LIGHT']], low_memory=False, header=None)
        light_dataframe.columns = ["timestamp", "value"]
        light_dataframe = light_dataframe.drop_duplicates()
        light_dataframe = light_dataframe.sort_values(by='timestamp')
        signif_motion_dataframe = pd.read_csv(filenames[data_sources_with_ids['ANDROID_SIGNIFICANT_MOTION']],
                                              low_memory=False, header=None)
        signif_motion_dataframe.columns = ["timestamp", "value"]
        signif_motion_dataframe = signif_motion_dataframe.drop_duplicates()
        signif_motion_dataframe = signif_motion_dataframe.sort_values(by='timestamp')
        step_detector_dataframe = pd.read_csv(filenames[data_sources_with_ids['ANDROID_STEP_DETECTOR']],
                                              low_memory=False,
                                              header=None)
        step_detector_dataframe.columns = ["timestamp", "value"]
        step_detector_dataframe = step_detector_dataframe.drop_duplicates()
        step_detector_dataframe = step_detector_dataframe.sort_values(by='timestamp')
        calls_dataframe = pd.read_csv(filenames[data_sources_with_ids['CALLS']], low_memory=False, header=None)
        calls_dataframe.columns = ["timestamp", "value"]
        calls_dataframe = calls_dataframe.drop_duplicates()
        calls_dataframe = calls_dataframe.sort_values(by='timestamp')
        sms_dataframe = pd.read_csv(filenames[data_sources_with_ids['SMS']], low_memory=False)
        sms_dataframe.columns = ["timestamp", "value"]
        sms_dataframe = sms_dataframe.drop_duplicates()
        sms_dataframe = sms_dataframe.sort_values(by='timestamp')
        notifications_dataframe = pd.read_csv(filenames[data_sources_with_ids['NOTIFICATIONS']], low_memory=False,
                                              header=None)
        notifications_dataframe.columns = ["timestamp", "value"]
        notifications_dataframe = notifications_dataframe.drop_duplicates()
        notifications_dataframe = notifications_dataframe.sort_values(by='timestamp')
        screen_state_dataframe = pd.read_csv(filenames[data_sources_with_ids['SCREEN_STATE']], low_memory=False,
                                             header=None)
        screen_state_dataframe.columns = ["timestamp", "value"]
        screen_state_dataframe = screen_state_dataframe.drop_duplicates()
        screen_state_dataframe = screen_state_dataframe.sort_values(by='timestamp')
        unlock_state_dataframe = pd.read_csv(filenames[data_sources_with_ids['UNLOCK_STATE']], low_memory=False,
                                             header=None)
        unlock_state_dataframe.columns = ["timestamp", "value"]
        unlock_state_dataframe = unlock_state_dataframe.drop_duplicates()
        unlock_state_dataframe = unlock_state_dataframe.sort_values(by='timestamp')
        microphone_dataframe = pd.read_csv(filenames[data_sources_with_ids['SOUND_DATA']], low_memory=False)
        microphone_dataframe.columns = ["timestamp", "value"]
        microphone_dataframe = microphone_dataframe.drop_duplicates()
        microphone_dataframe = microphone_dataframe.sort_values(by='timestamp')
        wifi_dataframe = pd.read_csv(filenames[data_sources_with_ids['ANDROID_WIFI']], low_memory=False, header=None)
        wifi_dataframe.columns = ["timestamp", "value"]
        wifi_dataframe = wifi_dataframe.drop_duplicates()
        wifi_dataframe = wifi_dataframe.sort_values(by='timestamp')
        typing_dataframe = pd.read_csv(filenames[data_sources_with_ids['TYPING']], low_memory=False, header=None)
        typing_dataframe.columns = ["timestamp", "value"]
        typing_dataframe = typing_dataframe.drop_duplicates()
        typing_dataframe = typing_dataframe.sort_values(by='timestamp')
        keystroke_log_dataframe = pd.read_csv(filenames[data_sources_with_ids['KEYSTROKE_LOG']], low_memory=False,
                                              header=None)
        keystroke_log_dataframe.columns = ["timestamp", "value"]
        keystroke_log_dataframe = keystroke_log_dataframe.drop_duplicates()
        keystroke_log_dataframe = keystroke_log_dataframe.sort_values(by='timestamp')
        gravity_dataframe = pd.read_csv(filenames[data_sources_with_ids['GRAVITY']], low_memory=False, header=None)
        gravity_dataframe.columns = ["timestamp", "value"]
        gravity_dataframe = gravity_dataframe.drop_duplicates()
        gravity_dataframe = gravity_dataframe.sort_values(by='timestamp')

        locations_gps_dataframe = pd.read_csv(filenames[data_sources_with_ids['LOCATION_GPS']], low_memory=False,
                                              header=None)
        locations_gps_dataframe.columns = ["timestamp", "value"]
        locations_gps_dataframe = locations_gps_dataframe.drop_duplicates()
        locations_gps_dataframe = locations_gps_dataframe.sort_values(by='timestamp')
        locations_manual_dataframe = pd.read_csv(filenames[data_sources_with_ids['LOCATIONS_MANUAL']], low_memory=False,
                                                 header=None)
        locations_manual_dataframe.columns = ["timestamp", "value"]
        locations_manual_dataframe = locations_manual_dataframe.drop_duplicates()
        locations_manual_dataframe = locations_manual_dataframe.sort_values(by='timestamp')

        # endregion

        for row in ema_table.itertuples():
            print('*************' + row.timestamp + '*************')
            ema_time_range = tools.get_ema_time_range_until_ema(int(row.timestamp))

            # region extracting features related to EMA

            for i, value in enumerate(ema_time_range['time_from']):
                print("Extracting activities features")
                activities_features = features_extraction.get_activity_recognition_features(activities_dataframe,
                                                                                            value,
                                                                                            ema_time_range['time_to'][
                                                                                                i])
                print("Extracting app usage features")
                app_usage_features = features_extraction.get_app_usage_features(app_usage_dataframe,
                                                                                value,
                                                                                ema_time_range['time_to'][i])
                print("Extracting light features")
                light_features = features_extraction.get_light_features(light_dataframe,
                                                                        value,
                                                                        ema_time_range['time_to'][i])
                print("Extracting significant motion features")
                significant_motion_features = features_extraction.get_signif_motion_features(signif_motion_dataframe,
                                                                                             value,
                                                                                             ema_time_range['time_to'][
                                                                                                 i])
                print("Extracting step detector features")
                step_detector_features = features_extraction.get_step_detector_features(step_detector_dataframe,
                                                                                        value,
                                                                                        ema_time_range['time_to'][i])
                print("Extracting calls features")
                calls_features = features_extraction.get_calls_features(calls_dataframe,
                                                                        value,
                                                                        ema_time_range['time_to'][i])
                print("Extracting sms features")
                sms_features = features_extraction.get_sms_features(sms_dataframe,
                                                                    value,
                                                                    ema_time_range['time_to'][i])
                print("Extracting notifications features")
                notifications_features = features_extraction.get_notifications_features(notifications_dataframe,
                                                                                        value,
                                                                                        ema_time_range['time_to'][i])
                print("Extracting screen state features")
                screen_state_features = features_extraction.get_screen_state_features(screen_state_dataframe,
                                                                                      value,
                                                                                      ema_time_range['time_to'][i])
                print("Extracting unlock state features")
                unlock_state_features = features_extraction.get_unlock_state_features(unlock_state_dataframe,
                                                                                      value,
                                                                                      ema_time_range['time_to'][i])
                print("Extracting microphone features")
                microphone_features = features_extraction.get_microphone_features(microphone_dataframe,
                                                                                  value,
                                                                                  ema_time_range['time_to'][i])
                print("Extracting wifi features")
                wifi_features = features_extraction.get_wifi_features(wifi_dataframe,
                                                                      value,
                                                                      ema_time_range['time_to'][i])
                print("Extracting typing features")
                typing_features = features_extraction.get_typing_features(typing_dataframe,
                                                                          value,
                                                                          ema_time_range['time_to'][i])
                print("Extracting keystroke log features")
                keystroke_log_features = features_extraction.get_keystroke_features(keystroke_log_dataframe,
                                                                                    value,
                                                                                    ema_time_range['time_to'][i])
                print("Extracting gravity features")
                gravity_features = features_extraction.get_gravity_features(gravity_dataframe,
                                                                            value,
                                                                            ema_time_range['time_to'][i])
                print("Extracting locations features")
                locations_features = features_extraction.get_locations_features(locations_gps_dataframe,
                                                                                locations_manual_dataframe,
                                                                                value,
                                                                                ema_time_range['time_to'][i])

                # endregion
                # region appending dataframe with extracted features
                extracted_features = {
                    'user_id': user_id,
                    'ema_timestamp': row.timestamp,
                    'phq1': row.phq1,
                    'phq2': row.phq2,
                    'phq3': row.phq3,
                    'phq4': row.phq4,
                    'phq5': row.phq5,
                    'phq6': row.phq6,
                    'phq7': row.phq7,
                    'phq8': row.phq8,
                    'phq9': row.phq9,
                    'still_freq': activities_features['still_freq'],
                    'walking_freq': activities_features['walking_freq'],
                    'running_freq': activities_features['running_freq'],
                    'on_bicycle_freq': activities_features['on_bicycle_freq'],
                    'in_vehicle_freq': activities_features['in_vehicle_freq'],
                    'still_dur': activities_features['still_dur'],
                    'walking_dur': activities_features['walking_dur'],
                    'running_dur': activities_features['running_dur'],
                    'on_bicycle_dur': activities_features['on_bicycle_dur'],
                    'in_vehicle_dur': activities_features['in_vehicle_dur'],
                    'app_entertainment_music_dur': app_usage_features['app_entertainment_music_dur'],
                    'app_utilities_dur': app_usage_features['app_utilities_dur'],
                    'app_shopping_dur': app_usage_features['app_shopping_dur'],
                    'app_games_comics_dur': app_usage_features['app_games_comics_dur'],
                    'app_others_dur': app_usage_features['app_others_dur'],
                    'app_health_wellness_dur': app_usage_features['app_health_wellness_dur'],
                    'app_social_communication_dur': app_usage_features['app_social_communication_dur'],
                    'app_education_dur': app_usage_features['app_education_dur'],
                    'app_travel_dur': app_usage_features['app_travel_dur'],
                    'app_art_design_photo_dur': app_usage_features['app_art_design_photo_dur'],
                    'app_news_magazine_dur': app_usage_features['app_news_magazine_dur'],
                    'app_food_drink_dur': app_usage_features['app_food_drink_dur'],
                    'app_unknown_background_dur': app_usage_features['app_unknown_background_dur'],
                    'app_entertainment_music_freq': app_usage_features['app_entertainment_music_freq'],
                    'app_utilities_freq': app_usage_features['app_utilities_freq'],
                    'app_shopping_freq': app_usage_features['app_shopping_freq'],
                    'app_games_comics_freq': app_usage_features['app_games_comics_freq'],
                    'app_others_freq': app_usage_features['app_others_freq'],
                    'app_health_wellness_freq': app_usage_features['app_health_wellness_freq'],
                    'app_social_communication_freq': app_usage_features['app_social_communication_freq'],
                    'app_education_freq': app_usage_features['app_education_freq'],
                    'app_travel_freq': app_usage_features['app_travel_freq'],
                    'app_art_design_photo_freq': app_usage_features['app_art_design_photo_freq'],
                    'app_news_magazine_freq': app_usage_features['app_news_magazine_freq'],
                    'app_food_drink_freq': app_usage_features['app_food_drink_freq'],
                    'app_unknown_background_freq': app_usage_features['app_unknown_background_freq'],
                    'apps_total_num': app_usage_features['apps_total_num'],
                    'apps_unique_num': app_usage_features['apps_unique_num'],
                    'browser_dur': app_usage_features['browser_dur'],
                    'light_min': light_features['light_min'],
                    'light_max': light_features['light_max'],
                    'light_avg': light_features['light_avg'],
                    'light_stddev': light_features['light_stddev'],
                    'light_dark_ratio': light_features['light_dark_ratio'],
                    'num_of_places': locations_features['num_of_places'],
                    'max_dur_at_place': locations_features['max_dur_at_place'],
                    'min_dur_at_place': locations_features['min_dur_at_place'],
                    'avg_dur_at_place': locations_features['avg_dur_at_place'],
                    'stdev_dur_at_place': locations_features['stdev_dur_at_place'],
                    'var_dur_at_place': locations_features['var_dur_at_place'],
                    'dur_of_homestay': locations_features['dur_of_homestay'],
                    'dur_at_work_study': locations_features['dur_at_work_study'],
                    'entropy': locations_features['entropy'],
                    'normalized_entropy': locations_features['normalized_entropy'],
                    'location_variance': locations_features['location_variance'],
                    'max_dist_from_home': locations_features['max_dist_from_home'],
                    'avg_dist_from_home': locations_features['avg_dist_from_home'],
                    'max_dist_btw_places': locations_features['max_dist_btw_places'],
                    'total_dist_travelled': locations_features['total_dist_travelled'],
                    'signif_motion_freq': significant_motion_features,
                    'steps_num': step_detector_features,
                    'calls_missed_num': calls_features['calls_missed_num'],
                    'calls_in_num': calls_features['calls_in_num'],
                    'calls_out_num': calls_features['calls_out_num'],
                    'calls_min_out_dur': calls_features['calls_min_out_dur'],
                    'calls_max_out_dur': calls_features['calls_max_out_dur'],
                    'calls_avg_out_dur': calls_features['calls_avg_out_dur'],
                    'calls_total_out_dur': calls_features['calls_total_out_dur'],
                    'calls_min_in_dur': calls_features['calls_min_in_dur'],
                    'calls_max_in_dur': calls_features['calls_max_in_dur'],
                    'calls_avg_in_dur': calls_features['calls_avg_in_dur'],
                    'calls_total_in_dur': calls_features['calls_total_in_dur'],
                    'sms_min_chars': sms_features['sms_min_chars'],
                    'sms_max_chars': sms_features['sms_max_chars'],
                    'sms_avg_chars': sms_features['sms_avg_chars'],
                    'sms_unique_contacts': sms_features['sms_unique_contacts'],
                    'sms_total_num': sms_features['sms_total_num'],
                    'notif_arrived_num': notifications_features['notif_arrived_num'],
                    'notif_clicked_num': notifications_features['notif_clicked_num'],
                    'notif_min_dec_time': notifications_features['notif_min_dec_time'],
                    'notif_max_dec_time': notifications_features['notif_max_dec_time'],
                    'notif_avg_dec_time': notifications_features['notif_avg_dec_time'],
                    'notif_stdev_dec_time': notifications_features['notif_stdev_dec_time'],
                    'screen_on_freq': screen_state_features['screen_on_freq'],
                    'screen_off_freq': screen_state_features['screen_off_freq'],
                    'lock_freq': unlock_state_features['lock_freq'],
                    'unlock_freq': unlock_state_features['unlock_freq'],
                    'unlock_dur': unlock_state_features['unlock_dur'],
                    'pitch_num': microphone_features['pitch_num'],
                    'pitch_avg': microphone_features['pitch_avg'],
                    'pitch_stdev': microphone_features['pitch_stdev'],
                    'sound_energy_min': microphone_features['sound_energy_min'],
                    'sound_energy_max': microphone_features['sound_energy_max'],
                    'sound_energy_avg': microphone_features['sound_energy_avg'],
                    'sound_energy_stdev': microphone_features['sound_energy_stdev'],
                    'wifi_unique_num': wifi_features,
                    'typing_freq': typing_features['typing_freq'],
                    'typing_unique_apps_num': typing_features['typing_unique_apps_num'],
                    'typing_min': typing_features['typing_min'],
                    'typing_max': typing_features['typing_max'],
                    'typing_avg': typing_features['typing_avg'],
                    'typing_stdev': typing_features['typing_stdev'],
                    'bkspace_ratio': keystroke_log_features['bkspace_ratio'],
                    'autocor_ratio': keystroke_log_features['autocor_ratio'],
                    'intrkey_delay_avg': keystroke_log_features['intrkey_delay_avg'],
                    'intrkey_delay_stdev': keystroke_log_features['intrkey_delay_stdev'],
                    'key_sessions_num': keystroke_log_features['key_sessions_num'],
                    'gr_x_mean': gravity_features['gr_x_mean'],
                    'gr_x_std': gravity_features['gr_x_std'],
                    'gr_y_mean': gravity_features['gr_y_mean'],
                    'gr_y_std': gravity_features['gr_y_std'],
                    'gr_z_mean': gravity_features['gr_z_mean'],
                    'gr_z_std': gravity_features['gr_z_std'],
                    'gender': user_ids_with_gender[user_id],
                    'weekday': tools.is_weekday(row.timestamp),
                    'depr_group': user_ids_with_depression_group[user_id]
                }
                # endregion

                output_table = output_table.append(extracted_features, ignore_index=True)
        output_table.to_csv(output_filename, index=False)
    return f'Feature extraction finished for {user_directory}'


def extract_features_double_period(user_directory):
    output_columns_double_period = [
        'user_id',
        'ema_timestamp',
        'images_dif',
        'videos_dif',
        'music_dif',
        'cal_events_dif'
    ]

    if user_directory != '.DS_Store':
        user_id = int(user_directory.split('-')[-1])
        filenames = tools.create_filenames(user_id, data_sources)
        output_table = pd.DataFrame(columns=output_columns_double_period)
        output_filename = f'features_until_EMA_double/extracted_features_{user_id}.csv'
        print(f'Feature extraction started for {user_directory}')

        ema_filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/54people/{user_directory}/{user_id}_11.csv'
        tools.remove_duplicate_ema(ema_filename)

        ema_table = pd.read_csv(ema_filename, delimiter=',', names=['timestamp', 'value'], header=None)
        ema_table = ema_table['value'].str.split(' ', n=10, expand=True)
        ema_table.columns = ['timestamp', 'ema_order', 'phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8',
                             'phq9']

        # region creating dataframes
        calendar_dataframe = pd.read_csv(filenames[data_sources_with_ids['CALENDAR']], low_memory=False, header=None)
        calendar_dataframe.columns = ["timestamp", "value"]
        calendar_dataframe = calendar_dataframe.drop_duplicates()
        calendar_dataframe = calendar_dataframe.sort_values(by='timestamp')

        stored_media_dataframe = pd.read_csv(filenames[data_sources_with_ids['STORED_MEDIA']], low_memory=False,
                                             header=None)
        stored_media_dataframe.columns = ["timestamp", "value"]
        stored_media_dataframe = stored_media_dataframe.drop_duplicates()
        stored_media_dataframe = stored_media_dataframe.sort_values(by='timestamp')

        # endregion

        for row in ema_table.itertuples():
            print('*************' + row.timestamp + '*************')
            ema_time_range = tools.get_ema_double_time_range_4hrs(int(row.timestamp))

            # region features extraction
            for i, value in enumerate(ema_time_range['time_from']):
                print("Extracting stored media features")
                stored_media_features = features_extraction.get_stored_media_features(stored_media_dataframe,
                                                                                      ema_time_range[
                                                                                          'prev_time_from'][i],
                                                                                      ema_time_range[
                                                                                          'prev_time_to'][i],
                                                                                      value,
                                                                                      ema_time_range['time_to'][i])

                print("Extracting calendar features")
                calendar_features = features_extraction.get_calendar_features(calendar_dataframe,
                                                                              ema_time_range[
                                                                                  'prev_time_from'][i],
                                                                              ema_time_range[
                                                                                  'prev_time_to'][i],
                                                                              value,
                                                                              ema_time_range['time_to'][i])
                # endregion

                # appending dataframe
                extracted_features = {
                    'user_id': user_id,
                    'ema_timestamp': row.timestamp,
                    'images_dif': stored_media_features['images_dif'],
                    'videos_dif': stored_media_features['videos_dif'],
                    'music_dif': stored_media_features['music_dif'],
                    'cal_events_dif': calendar_features
                }

                output_table = output_table.append(extracted_features, ignore_index=True)

        output_table.to_csv(output_filename, index=False)
    return f'Feature extraction finished for {user_directory}'


def add_sleep_duration(user_directory):
    if user_directory != '.DS_Store':
        # user_directory = f"4-{user_directory.split('_')[-1].split('.')[0]}"
        user_id = int(user_directory.split('-')[-1])
        print('Sleep extraction started for user', user_id)

        output_filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/features_until_EMA/extracted_features_{user_id}.csv'
        app_usage_filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/54people/{user_directory}/{user_id}_28.csv'
        ema_filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/54people/{user_directory}/{user_id}_11.csv'

        ema_table = pd.read_csv(ema_filename, delimiter=',', names=['timestamp', 'value'], header=None)
        ema_table = ema_table['value'].str.split(' ', n=10, expand=True)
        ema_table.columns = ['timestamp', 'ema_order', 'phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8',
                             'phq9']

        app_usage_dataframe = pd.read_csv(app_usage_filename, low_memory=False,
                                          header=None)
        app_usage_dataframe.columns = ["timestamp", "value"]
        app_usage_dataframe.drop_duplicates()
        app_usage_dataframe.sort_values(by='timestamp')

        output_dataframe = pd.read_csv(output_filename)

        sleep_features_all = features_extraction.get_sleep_duration(app_usage_dataframe)
        sleep_features = {
            'sleep_dur': [],
            'sleep_start': [],
            'sleep_end': []
        }
        for row in output_dataframe.itertuples():
            print('*************' + str(row.ema_timestamp) + '*************')
            timestamp_date = datetime.fromtimestamp(int(row.ema_timestamp) / 1000).date()
            try:
                sleep_features['sleep_dur'].append(sleep_features_all[timestamp_date][0])
                sleep_features['sleep_start'].append(sleep_features_all[timestamp_date][1])
                sleep_features['sleep_end'].append(sleep_features_all[timestamp_date][2])
            except:
                sleep_features['sleep_dur'].append(np.nan)
                sleep_features['sleep_start'].append(np.nan)
                sleep_features['sleep_end'].append(np.nan)

        output_dataframe['sleep_dur'] = sleep_features['sleep_dur']
        output_dataframe['sleep_start'] = sleep_features['sleep_start']
        output_dataframe['sleep_end'] = sleep_features['sleep_end']

        output_dataframe.to_csv(output_filename, index=False)

    return f'Sleep extraction finished for user: {user_directory}'


def add_weather_data(user_directory):
    # for each user separately
    if user_directory != '.DS_Store':
        # user_directory = f"4-{user_directory.split('_')[-1].split('.')[0]}"
        print(f'Weather features extraction started for {user_directory}')
        weather_df = pd.read_csv('/Users/aliceberg/Programming/PyCharm/STDD-FE/incheon.csv', index_col='date_time')
        user_id = int(user_directory.split('-')[-1])

        output_filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/features_until_EMA/extracted_features_{user_id}.csv'
        ema_filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/54people/{user_directory}/{user_id}_11.csv'

        ema_table = pd.read_csv(ema_filename, delimiter=',', names=['timestamp', 'value'], header=None)
        ema_table = ema_table['value'].str.split(' ', n=10, expand=True)
        ema_table.columns = ['timestamp', 'ema_order', 'phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8',
                             'phq9']

        output_dataframe = pd.read_csv(output_filename)

        weather_features = {
            'tempC': [],
            'totalSnow_cm': [],
            'cloudcover': [],
            'precipMM': [],
            'windspeedKmph': [],
        }

        for row in output_dataframe.itertuples():
            print('*************' + str(row.ema_timestamp) + '*************')
            timestamp_year = tools.from_timestamp_to_year(row.ema_timestamp)
            timestamp_month = tools.from_timestamp_to_month(row.ema_timestamp)
            timestamp_day = tools.from_timestamp_to_day(row.ema_timestamp)
            timestamp_hour = tools.from_timestamp_to_hour(row.ema_timestamp)

            date_time = str(timestamp_year) + '-'
            # region converting timestamp to the right date format
            if timestamp_month < 10:
                date_time += '0' + str(timestamp_month) + '-'
            else:
                date_time += str(timestamp_month) + '-'

            if timestamp_day < 10:
                date_time += '0' + str(timestamp_day) + ' '
            else:
                date_time += str(timestamp_day) + ' '

            if timestamp_hour == 10:
                date_time += '09:00:00'
            elif timestamp_hour == 18:
                date_time += '18:00:00'
            elif timestamp_hour == 22:
                date_time += '21:00:00'
            elif timestamp_hour == 14:
                date_time += '12:00:00'
            # endregion

            weather_features['tempC'].append(weather_df.loc[date_time].tempC)
            weather_features['totalSnow_cm'].append(weather_df.loc[date_time].totalSnow_cm)
            weather_features['cloudcover'].append(weather_df.loc[date_time].cloudcover)
            weather_features['precipMM'].append(weather_df.loc[date_time].precipMM)
            weather_features['windspeedKmph'].append(weather_df.loc[date_time].windspeedKmph)

        output_dataframe['tempC'] = weather_features['tempC']
        output_dataframe['totalSnow_cm'] = weather_features['totalSnow_cm']
        output_dataframe['cloudcover'] = weather_features['cloudcover']
        output_dataframe['precipMM'] = weather_features['precipMM']
        output_dataframe['windspeedKmph'] = weather_features['windspeedKmph']

        output_dataframe.to_csv(output_filename, index=False)

    return f'Weather features extraction finished for {user_directory}'


def missing_data_imputation_ffill(user_directory):
    if user_directory != '.DS_Store':
        print(f'Started missing data imputation for {user_directory}')
        # user_id = int(user_directory.split('-')[-1])
        # output_filename = f'/Users/aliceberg/Programming/PyCharm/STDD-FE/extracted_features2/extracted_features_{user_id}.csv'
        output_filename = user_directory
        output_dataframe = pd.read_csv(output_filename)
        columns = ['sound_energy_min',
                   'sound_energy_max', 'sound_energy_avg', 'sound_energy_stdev']
        for col in columns:
            output_dataframe[col] = output_dataframe[col].ffill()

        output_dataframe.to_csv(output_filename, index=False)

    return f'Finished missing data imputation for {user_directory}'


def social_act_score_calculation(filename):
    i = 3
    df_main = pd.read_csv(filename)
    columns = ['app_social_communication_dur',
               'app_social_communication_freq',
               'calls_in_num',
               'calls_out_num',
               'calls_min_out_dur',
               'calls_max_out_dur',
               'calls_avg_out_dur',
               'calls_total_out_dur',
               'calls_min_in_dur',
               'calls_max_in_dur',
               'calls_avg_in_dur',
               'calls_total_in_dur',
               'sms_min_chars',
               'sms_max_chars',
               'sms_avg_chars',
               'sms_unique_contacts',
               'sms_total_num',
               'total_dist_travelled']

    # splitting by user_id
    gb = df_main.groupby('user_id')
    [gb.get_group(x) for x in gb.groups]

    social_val = []
    social_score = []
    results_row = []
    mean = {}
    sd = {}

    for col in columns:
        mean[col] = df_main[col].mean()
        sd[col] = df_main[col].std(skipna=True)

    for user_id in gb.groups:
        df = gb.get_group(user_id)

        for index, row in df.iterrows():
            for col in columns:
                r = ((((row[col] - mean[col]) / sd[col]) + i) / (2 * i)) * 100
                if not math.isnan(r):
                    results_row.append(r)

            result = sum(results_row) / len(results_row)
            results_row = []
            social_val.append(result)

            if result < 20:
                social_score.append(1)
            elif 20 <= result < 40:
                social_score.append(2)
            elif 40 <= result < 60:
                social_score.append(3)
            elif 60 <= result < 80:
                social_score.append(4)
            elif 80 <= result <= 100:
                social_score.append(5)
            else:
                social_score.append(0)

    df_main['social_val'] = social_val
    df_main['social_score'] = social_score

    df_main.to_csv(filename, index=False)


def sleep_score_calculation(filename):
    df = pd.read_csv(filename)
    sleep_scores = []

    for row in df.itertuples():
        if not np.isnan(row.sleep_dur):
            sleep_hours = int(row.sleep_dur) / 60
            if sleep_hours < 4 or sleep_hours > 11:
                sleep_scores.append(1)
            elif 4 <= sleep_hours < 5 or 10 < sleep_hours <= 11:
                sleep_scores.append(2)
            elif 5 <= sleep_hours < 6 or 9 < sleep_hours <= 10:
                sleep_scores.append(3)
            elif 6 <= sleep_hours < 7 or 8 < sleep_hours <= 9:
                sleep_scores.append(4)
            elif 7 <= sleep_hours <= 8:
                sleep_scores.append(5)
        else:
            sleep_scores.append(np.nan)

    df['sleep_score'] = sleep_scores
    df.to_csv(filename, index=False)


def convert_ema_to_symptom_scores(filename):
    dataframe = pd.read_csv(filename)

    mood_scores = []
    food_scores = []
    sleep_scores = []
    physical_activity_scores = []
    social_activity_scores = []

    for row in dataframe.itertuples():
        # re-init
        mood_ema = []
        food_ema = []
        sleep_ema = []
        physical_activity_ema = []
        social_activity_ema = []

        mood_ema.append(row.phq1)
        mood_ema.append(row.phq7)
        mood_ema.append(row.phq9)

        food_ema.append(row.phq3)

        sleep_ema.append(row.phq4)

        physical_activity_ema.append(row.phq5)
        physical_activity_ema.append(row.phq6)

        social_activity_ema.append(row.phq2)
        social_activity_ema.append(row.phq8)

        mood_score = round(statistics.mean(mood_ema))
        food_score = food_ema[0]
        sleep_score = sleep_ema[0]
        physical_activity_score = round(statistics.mean(physical_activity_ema))
        social_activity_score = round(statistics.mean(social_activity_ema))

        # region converting scores
        if mood_score == 5:
            mood_score = 1
        elif mood_score == 4:
            mood_score = 2
        elif mood_score == 3:
            mood_score = 3
        elif mood_score == 2:
            mood_score = 4
        elif mood_score == 1:
            mood_score = 5

        if food_score == 5:
            food_score = 1
        elif food_score == 4:
            food_score = 2
        elif food_score == 3:
            food_score = 3
        elif food_score == 2:
            food_score = 4
        elif food_score == 1:
            food_score = 5

        if sleep_score == 5:
            sleep_score = 1
        elif sleep_score == 4:
            sleep_score = 2
        elif sleep_score == 3:
            sleep_score = 3
        elif sleep_score == 2:
            sleep_score = 4
        elif sleep_score == 1:
            sleep_score = 5

        if social_activity_score == 5:
            social_activity_score = 1
        elif social_activity_score == 4:
            social_activity_score = 2
        elif social_activity_score == 3:
            social_activity_score = 3
        elif social_activity_score == 2:
            social_activity_score = 4
        elif social_activity_score == 1:
            social_activity_score = 5

        if physical_activity_score == 5:
            physical_activity_score = 1
        elif physical_activity_score == 4:
            physical_activity_score = 2
        elif physical_activity_score == 3:
            physical_activity_score = 3
        elif physical_activity_score == 2:
            physical_activity_score = 4
        elif physical_activity_score == 1:
            physical_activity_score = 5

        # endregion

        mood_scores.append(mood_score)
        food_scores.append(food_score)
        sleep_scores.append(sleep_score)
        physical_activity_scores.append(physical_activity_score)
        social_activity_scores.append(social_activity_score)

    # adding new columns to dataframe
    dataframe['mood_gt'] = mood_scores
    dataframe['food_gt'] = food_scores
    dataframe['sleep_gt'] = sleep_scores
    dataframe['physical_act_gt'] = physical_activity_scores
    dataframe['social_act_gt'] = social_activity_scores

    dataframe.to_csv(filename, index=False)


def remove_missing_values_rows(filename, threshold):
    df = pd.read_csv(filename, low_memory=False)
    df = df[df.isnull().sum(axis=1) < threshold]

    df.to_csv(filename, index=False)


def main():
    start = time.perf_counter()
    print('Started')

    # combining sms files
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(tools.combine_sms_files, filename) for filename in os.listdir(data_directory)]

    for f in concurrent.futures.as_completed(results):
        print(f.result())

    finish = time.perf_counter()
    print(f'Finished in {round(finish - start)} second(s)')


if __name__ == '__main__':
    main()
