import concurrent.futures
import os
import time

import pandas as pd

import features_extraction
from tools import create_filenames, get_ema_time_range, is_weekday

USER_ID = 89
directory = 'data_for_fe'
data_sources = [1, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 71]

user_ids = [86, 87, 89, 90, 91, 92, 93, 100, 102, 125, 119, 110]

user_ids_with_gender = {
    '86': 'F',
    '87': 'M',
    '89': 'F',
    '90': 'M',
    '91': 'M',
    '92': 'M',
    '93': 'F',
    '100': 'M',
    '102': 'F',
    '125': 'F',
    '119': 'M',
    '110': 'F'

}
user_ids_with_depression_group = {
    '86': 'B',
    '87': 'A',
    '89': 'A',
    '90': 'C',
    '91': 'A',
    '92': 'B',
    '93': 'B',
    '100': 'B',
    '102': 'A',
    '125': 'C',
    '119': 'C',
    '110': 'C'

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
    'light_min',
    'light_max',
    'light_avg',
    'light_stddev',
    'light_dark_ratio',
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
    'pitch_num',
    'pitch_min',
    'pitch_max',
    'pitch_avg',
    'pitch_stdev',
    'sound_energy_min',
    'sound_energy_max',
    'sound_energy_avg',
    'sound_energy_stdev',
    'images_num',
    'videos_num',
    'music_num',
    'wifi_unique_num',
    'typing_freq',
    'typing_unique_apps_num',
    'typing_min',
    'typing_max',
    'typing_avg',
    'typing_stdev',
    'cal_events_num',
    'gender',
    'weekday',  # 1 if weekday, 0 otherwise
    'depr_group'

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


def extract_features(user_directory, user_id):
    filenames = create_filenames(user_id, data_sources)
    output_table = pd.DataFrame(columns=output_columns)
    print(f'Feature extraction started for {user_directory}')
    ema_filename = f'{user_directory}/{user_id}_11.csv'

    ema_table = pd.read_csv(ema_filename, delimiter=',', names=['timestamp', 'value'])
    ema_table = ema_table['value'].str.split(' ', n=10, expand=True)
    ema_table.columns = ['timestamp', 'ema_order', 'phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8',
                         'phq9']

    # region creating dataframes
    activities_dataframe = pd.read_csv(filenames[data_sources_with_ids['ACTIVITY_RECOGNITION']], low_memory=False)
    activities_dataframe.columns = ["timestamp", "value"]
    activities_dataframe.drop_duplicates()
    app_usage_dataframe = pd.read_csv(filenames[data_sources_with_ids['APPLICATION_USAGE']], low_memory=False)
    app_usage_dataframe.columns = ["timestamp", "value"]
    app_usage_dataframe.drop_duplicates()
    light_dataframe = pd.read_csv(filenames[data_sources_with_ids['ANDROID_LIGHT']], low_memory=False)
    light_dataframe.columns = ["timestamp", "value"]
    light_dataframe.drop_duplicates()
    signif_motion_dataframe = pd.read_csv(filenames[data_sources_with_ids['ANDROID_SIGNIFICANT_MOTION']],
                                          low_memory=False)
    signif_motion_dataframe.columns = ["timestamp", "value"]
    signif_motion_dataframe.drop_duplicates()
    step_detector_dataframe = pd.read_csv(filenames[data_sources_with_ids['ANDROID_STEP_DETECTOR']], low_memory=False)
    step_detector_dataframe.columns = ["timestamp", "value"]
    step_detector_dataframe.drop_duplicates()
    calls_dataframe = pd.read_csv(filenames[data_sources_with_ids['CALLS']], low_memory=False)
    calls_dataframe.columns = ["timestamp", "value"]
    calls_dataframe.drop_duplicates()
    sms_dataframe = pd.read_csv(filenames[data_sources_with_ids['SMS']], low_memory=False)
    sms_dataframe.columns = ["timestamp", "value"]
    sms_dataframe.drop_duplicates()
    notifications_dataframe = pd.read_csv(filenames[data_sources_with_ids['NOTIFICATIONS']], low_memory=False)
    notifications_dataframe.columns = ["timestamp", "value"]
    notifications_dataframe.drop_duplicates()
    screen_state_dataframe = pd.read_csv(filenames[data_sources_with_ids['SCREEN_STATE']], low_memory=False)
    screen_state_dataframe.columns = ["timestamp", "value"]
    screen_state_dataframe.drop_duplicates()
    unlock_state_dataframe = pd.read_csv(filenames[data_sources_with_ids['UNLOCK_STATE']], low_memory=False)
    unlock_state_dataframe.columns = ["timestamp", "value"]
    unlock_state_dataframe.drop_duplicates()
    microphone_dataframe = pd.read_csv(filenames[data_sources_with_ids['SOUND_DATA']], low_memory=False)
    microphone_dataframe.columns = ["timestamp", "value"]
    microphone_dataframe.drop_duplicates()
    stored_media_dataframe = pd.read_csv(filenames[data_sources_with_ids['STORED_MEDIA']], low_memory=False)
    stored_media_dataframe.columns = ["timestamp", "value"]
    stored_media_dataframe.drop_duplicates()
    wifi_dataframe = pd.read_csv(filenames[data_sources_with_ids['ANDROID_WIFI']], low_memory=False)
    wifi_dataframe.columns = ["timestamp", "value"]
    wifi_dataframe.drop_duplicates()
    typing_dataframe = pd.read_csv(filenames[data_sources_with_ids['TYPING']], low_memory=False)
    typing_dataframe.columns = ["timestamp", "value"]
    typing_dataframe.drop_duplicates()
    locations_gps_dataframe = pd.read_csv(filenames[data_sources_with_ids['LOCATION_GPS']], low_memory=False)
    locations_gps_dataframe.columns = ["timestamp", "value"]
    locations_gps_dataframe.drop_duplicates()
    locations_manual_dataframe = pd.read_csv(filenames[data_sources_with_ids['LOCATIONS_MANUAL']], low_memory=False)
    locations_manual_dataframe.columns = ["timestamp", "value"]
    locations_manual_dataframe.drop_duplicates()
    calendar_dataframe = pd.read_csv(filenames[data_sources_with_ids['CALENDAR']], low_memory=False)
    calendar_dataframe.columns = ["timestamp", "value"]
    calendar_dataframe.drop_duplicates()

    # endregion

    for row in ema_table.itertuples():
        ema_time_range = get_ema_time_range(int(row.timestamp))
        # region extracting features related to EMA
        activities_features = features_extraction.get_activity_recognition_features(activities_dataframe,
                                                                                    ema_time_range['time_from'],
                                                                                    ema_time_range['time_to'])
        app_usage_features = features_extraction.get_activity_recognition_features(app_usage_dataframe,
                                                                                   ema_time_range['time_from'],
                                                                                   ema_time_range['time_to'])
        light_features = features_extraction.get_light_features(light_dataframe,
                                                                ema_time_range['time_from'],
                                                                ema_time_range['time_to'])
        significant_motion_features = features_extraction.get_signif_motion_features(signif_motion_dataframe,
                                                                                     ema_time_range['time_from'],
                                                                                     ema_time_range['time_to'])
        step_detector_features = features_extraction.get_step_detector_features(step_detector_dataframe,
                                                                                ema_time_range['time_from'],
                                                                                ema_time_range['time_to'])
        calls_features = features_extraction.get_calls_features(calls_dataframe,
                                                                ema_time_range['time_from'],
                                                                ema_time_range['time_to'])
        sms_features = features_extraction.get_sms_features(sms_dataframe,
                                                            ema_time_range['time_from'],
                                                            ema_time_range['time_to'])
        notifications_features = features_extraction.get_notifications_features(notifications_dataframe,
                                                                                ema_time_range['time_from'],
                                                                                ema_time_range['time_to'])
        screen_state_features = features_extraction.get_screen_state_features(screen_state_dataframe,
                                                                              ema_time_range['time_from'],
                                                                              ema_time_range['time_to'])
        unlock_state_features = features_extraction.get_unlock_state_features(unlock_state_dataframe,
                                                                              ema_time_range['time_from'],
                                                                              ema_time_range['time_to'])
        microphone_features = features_extraction.get_microphone_features(microphone_dataframe,
                                                                          ema_time_range['time_from'],
                                                                          ema_time_range['time_to'])
        stored_media_features = features_extraction.get_stored_media_features(stored_media_dataframe,
                                                                              ema_time_range['time_from'],
                                                                              ema_time_range['time_to'])
        wifi_features = features_extraction.get_wifi_features(wifi_dataframe,
                                                              ema_time_range['time_from'],
                                                              ema_time_range['time_to'])
        typing_features = features_extraction.get_typing_features(typing_dataframe,
                                                                  ema_time_range['time_from'],
                                                                  ema_time_range['time_to'])
        calendar_features = features_extraction.get_calendar_features(calendar_dataframe,
                                                                      ema_time_range['time_from'],
                                                                      ema_time_range['time_to'])
        locations_features = features_extraction.get_locations_features(locations_gps_dataframe,
                                                                        ema_time_range['time_from'],
                                                                        ema_time_range['time_to'])

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
            'light_min': light_features['light_min'],
            'light_max': light_features['light_max'],
            'light_avg': light_features['light_avg'],
            'light_stddev': light_features['light_stddev'],
            'light_dark_ratio': light_features['light_dark_ratio'],
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
            'pitch_num': microphone_features['pitch_num'],
            'pitch_min': microphone_features['pitch_min'],
            'pitch_max': microphone_features['pitch_max'],
            'pitch_avg': microphone_features['pitch_avg'],
            'pitch_stdev': microphone_features['pitch_stdev'],
            'sound_energy_min': microphone_features['sound_energy_min'],
            'sound_energy_max': microphone_features['sound_energy_max'],
            'sound_energy_avg': microphone_features['sound_energy_avg'],
            'sound_energy_stdev': microphone_features[ 'sound_energy_stdev'],
            'images_num': stored_media_features['images_num'],
            'videos_num': stored_media_features['videos_num'],
            'music_num': stored_media_features['music_num'],
            'wifi_unique_num': wifi_features,
            'typing_freq': typing_features['typing_freq'],
            'typing_unique_apps_num': typing_features['typing_unique_apps_num'],
            'typing_min': typing_features['typing_min'],
            'typing_max': typing_features['typing_max'],
            'typing_avg': typing_features['typing_avg'],
            'typing_stdev': typing_features['typing_stdev'],
            'cal_events_num': calendar_features,
            'gender': user_ids_with_gender[user_id],
            'weekday': is_weekday(row.timestamp),
            'depr_group': user_ids_with_depression_group[user_id]
        }
        # endregion

        return f'Feature extraction finished for {user_directory}'

    def main():
        start = time.perf_counter()
        # can be done in parallel only per participants and not per data sources
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(extract_features, filename) for filename in os.listdir(directory)]

        for f in concurrent.futures.as_completed(results):
            print(f.result())

        finish = time.perf_counter()
        print(f'Finished in {round(finish - start)} second(s)')

    if __name__ == '__main__':
        main()
