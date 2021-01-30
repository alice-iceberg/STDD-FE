import pickle

import numpy as np
import pandas as pd
import xgboost
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_and_save_physical_act_models(filename):
    df = pd.read_csv(filename)

    # splitting by user_id
    gb = df.groupby('user_id')
    [gb.get_group(x) for x in gb.groups]

    for user_id in gb.groups:
        accuracies_output_mean = []
        accuracies_output_std = []
        print(f'Physical activity model for {user_id}')
        output_filename = '/Users/aliceberg/Programming/PyCharm/STDD-FE/physical_act/' + str(
            user_id) + '_physical_act.pkl'

        df = gb.get_group(user_id)
        X = df.iloc[:, 2:-1].values
        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)

        accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
        accuracies_output_mean.append(accuracies.mean() * 100)
        accuracies_output_std.append(accuracies.std() * 100)

        print('Mean accuracy for user ', user_id, 'is: ', accuracies_output_mean[len(accuracies_output_mean) - 1])

        with open(output_filename, 'wb+') as f:
            pickle.dump(classifier, f)

        with open('physical_act_train_accuracies.txt', 'a+') as file:
            line = '**************************************************\n\n'
            file.write(line)
            line = 'Mean accuracy for user ' + str(user_id) + ' is: ' + str(
                accuracies_output_mean[len(accuracies_output_mean) - 1]) + '\n'
            file.write(line)
            line = 'Details:\n'
            file.write(line)
            line = 'Mean accuracies:' + str(accuracies_output_mean) + '\n'
            file.write(line)
            line = 'Std: ' + str(accuracies_output_std) + '\n'
            file.write(line)


def train_and_save_mood_models(filename):
    df = pd.read_csv(filename)

    # splitting by user_id
    gb = df.groupby('user_id')
    [gb.get_group(x) for x in gb.groups]

    for user_id in gb.groups:
        accuracies_output_mean = []
        accuracies_output_std = []
        output_filename = '/Users/aliceberg/Programming/PyCharm/STDD-FE/mood/' + str(
            user_id) + '_mood.pkl'

        df = gb.get_group(user_id)
        X = df.iloc[:, 2:-1].values
        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)

        accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
        accuracies_output_mean.append(accuracies.mean() * 100)
        accuracies_output_std.append(accuracies.std() * 100)

        print('Mean accuracy for user ', user_id, 'is: ', accuracies_output_mean[len(accuracies_output_mean) - 1])

        with open(output_filename, 'wb+') as f:
            pickle.dump(classifier, f)

        with open('mood_train_accuracies.txt', 'a+') as file:
            line = '**************************************************\n\n'
            file.write(line)
            line = 'Mean accuracy for user ' + str(user_id) + ' is: ' + str(
                accuracies_output_mean[len(accuracies_output_mean) - 1]) + '\n'
            file.write(line)
            line = 'Details:\n'
            file.write(line)
            line = 'Mean accuracies:' + str(accuracies_output_mean) + '\n'
            file.write(line)
            line = 'Std: ' + str(accuracies_output_std) + '\n'
            file.write(line)


def predict_physical_act_and_mood(filename):
    pred_mood = []
    pred_phy_act = []
    main_df = pd.read_csv(filename)

    for row in main_df.itertuples():
        physical_model_filename = '/Users/aliceberg/Programming/PyCharm/STDD-FE/physical_act/' + str(
            row.user_id) + '_physical_act.pkl'
        mood_model_filename = '/Users/aliceberg/Programming/PyCharm/STDD-FE/mood/' + str(
            row.user_id) + '_mood.pkl'

        with open(physical_model_filename, 'rb') as f:
            clf = pickle.load(f)

            X_physical = [
                row.still_freq,
                row.walking_freq,
                row.running_freq,
                row.on_bicycle_freq,
                row.in_vehicle_freq,
                row.signif_motion_freq,
                row.steps_num,
                row.weekday,
                row.gender]

            X_trans_physical = np.array(X_physical).reshape((1, -1))
            physical_pred = clf.predict(X_trans_physical)
            pred_phy_act.append(int(physical_pred))

        with open(mood_model_filename, 'rb') as f:
            clf = pickle.load(f)
            X_mood = [
                row.still_freq,
                row.walking_freq,
                row.running_freq,
                row.on_bicycle_freq,
                row.in_vehicle_freq,
                row.signif_motion_freq,
                row.steps_num,
                row.app_entertainment_music_dur,
                row.app_utilities_dur,
                row.app_shopping_dur,
                row.app_games_comics_dur,
                row.app_others_dur,
                row.app_health_wellness_dur,
                row.app_social_communication_dur,
                row.app_education_dur,
                row.app_travel_dur,
                row.app_art_design_photo_dur,
                row.app_news_magazine_dur,
                row.app_food_drink_dur,
                row.app_unknown_background_dur,
                row.app_entertainment_music_freq,
                row.app_utilities_freq,
                row.app_shopping_freq,
                row.app_games_comics_freq,
                row.app_others_freq,
                row.app_health_wellness_freq,
                row.app_social_communication_freq,
                row.app_education_freq,
                row.app_travel_freq,
                row.app_art_design_photo_freq,
                row.app_news_magazine_freq,
                row.app_food_drink_freq,
                row.app_unknown_background_freq,
                row.apps_total_num,
                row.apps_unique_num,
                row.light_min,
                row.light_max,
                row.light_avg,
                row.light_stddev,
                row.light_dark_ratio,
                row.notif_arrived_num,
                row.notif_clicked_num,
                row.notif_min_dec_time,
                row.notif_max_dec_time,
                row.notif_avg_dec_time,
                row.notif_stdev_dec_time,
                row.screen_on_freq,
                row.screen_off_freq,
                row.lock_freq,
                row.unlock_freq,
                row.pitch_num,
                row.pitch_min,
                row.pitch_max,
                row.pitch_avg,
                row.pitch_stdev,
                row.sound_energy_min,
                row.sound_energy_max,
                row.sound_energy_avg,
                row.sound_energy_stdev,
                row.images_num,
                row.videos_num,
                row.music_num,
                row.wifi_unique_num,
                row.typing_freq,
                row.typing_unique_apps_num,
                row.typing_max,
                row.typing_avg,
                row.typing_stdev,
                row.cal_events_num,
                row.tempC,
                row.totalSnow_cm,
                row.cloudcover,
                row.precipMM,
                row.windspeedKmph,
                row.weekday,
                row.gender,
            ]

            X_trans_mood = np.array(X_mood).reshape((1, -1))
            prediction_mood = clf.predict(X_trans_mood)
            pred_mood.append(int(prediction_mood))

    main_df['mood_pred'] = pred_mood
    main_df['phy_act_pred'] = pred_phy_act
    main_df.to_csv('features_with_clusters.csv', index=False)


def create_symptom_clusters_file(filename):
    df = pd.read_csv(filename)
    symptom_clusters_df = pd.DataFrame()

    symptom_clusters_df['user_id'] = df['user_id']
    symptom_clusters_df['ema_timestamp'] = df['ema_timestamp']
    symptom_clusters_df['mood'] = df['mood_pred']
    symptom_clusters_df['physical_act'] = df['phy_act_pred']
    symptom_clusters_df['sleep'] = df['sleep_score']
    symptom_clusters_df['social_act'] = df['social_score']
    symptom_clusters_df['food'] = df['food_gt']
    symptom_clusters_df['depr_group'] = df['depr_group']

    symptom_clusters_df.to_csv('symptom_clusters.csv', index=False)


def get_physical_act_features_importance():
    user_ids = [86, 87, 89, 90, 91, 92, 93, 100, 102, 119, 110, 125]
    for user_id in user_ids:
        figure_name = 'physical_features(gain)' + str(user_id) + '.png'
        physical_model_filename = '/Users/aliceberg/Programming/PyCharm/STDD-FE/physical_act/' + str(
            user_id) + '_physical_act.pkl'
        with open(physical_model_filename, 'rb') as f:
            clf = pickle.load(f)
            clf.get_booster().feature_names = ['still_freq',
                                               'walking_freq',
                                               'running_freq',
                                               'on_bicycle_freq',
                                               'in_vehicle_freq',
                                               'signif_motion_freq',
                                               'steps_num',
                                               'weekday',
                                               'gender']
            xgboost.plot_importance(clf.get_booster(), importance_type='gain', max_num_features=15,
                                    show_values=False)
            pyplot.savefig(figure_name, bbox_inches='tight')
            # with open('physical_features_importance.txt', 'a+') as file:
            #     file.write(str(line))
            #     file.write('\n')


def get_mood_features_importance(user_ids):
    for user_id in user_ids:
        physical_model_filename = '/Users/aliceberg/Programming/PyCharm/STDD-FE/mood/' + str(user_id) + '_mood.pkl'
        with open(physical_model_filename, 'rb') as f:
            clf = pickle.load(f)
            clf.get_booster().feature_names = ['still_freq',
                                               'walking_freq',
                                               'running_freq',
                                               'on_bicycle_freq',
                                               'in_vehicle_freq',
                                               'signif_motion_freq',
                                               'steps_num',
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
                                               'typing_max',
                                               'typing_avg',
                                               'typing_stdev',
                                               'cal_events_num',
                                               'tempC',
                                               'totalSnow_cm',
                                               'cloudcover',
                                               'precipMM',
                                               'windspeedKmph',
                                               'weekday',
                                               'gender']
            line = clf.get_booster().get_score(importance_type='gain')
            with open('mood_features_importance.txt', 'a+') as file:
                file.write(str(line))
                file.write('\n\n')


def plot_mood_features_importance():
    user_ids = [87, 89, 90, 91, 92, 93, 100, 102, 119, 110, 125]
    for user_id in user_ids:
        figure_name = 'mood_features(gain)' + str(user_id) + '.png'
        mood_model_name = '/Users/aliceberg/Programming/PyCharm/STDD-FE/mood/' + str(user_id) + '_mood.pkl'
        with open(mood_model_name, 'rb') as f:
            clf = pickle.load(f)
            clf.get_booster().feature_names = ['still_freq',
                                               'walking_freq',
                                               'running_freq',
                                               'on_bicycle_freq',
                                               'in_vehicle_freq',
                                               'signif_motion_freq',
                                               'steps_num',
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
                                               'typing_max',
                                               'typing_avg',
                                               'typing_stdev',
                                               'cal_events_num',
                                               'tempC',
                                               'totalSnow_cm',
                                               'cloudcover',
                                               'precipMM',
                                               'windspeedKmph',
                                               'weekday',
                                               'gender']
            xgboost.plot_importance(clf.get_booster(), importance_type='gain', max_num_features=15,
                                    show_values=False)
            pyplot.savefig(figure_name, bbox_inches='tight')


def plot_physical_act_features_importance():
    user_ids = [86]
    for user_id in user_ids:
        physical_model_filename = '/Users/aliceberg/Programming/PyCharm/STDD-FE/physical_act/' + str(
            user_id) + '_physical_act.pkl'
        with open(physical_model_filename, 'rb') as f:
            clf = pickle.load(f)
            clf.get_booster().feature_names = ['still_freq',
                                               'walking_freq',
                                               'running_freq',
                                               'on_bicycle_freq',
                                               'in_vehicle_freq',
                                               'signif_motion_freq',
                                               'steps_num',
                                               'weekday',
                                               'gender']
            xgboost.plot_importance(clf.get_booster(), importance_type='total_gain', show_values=False)
            pyplot.savefig('physical_act_features(total_gain)B.png', bbox_inches='tight')


def plot_features_importance_total():
    filename = '/Users/aliceberg/Programming/PyCharm/STDD-FE/depr_mode.pkl'
    with open(filename, 'rb') as f:
        classifier = pickle.load(f)
        classifier.get_booster().feature_names = ['mood',
                                                  'physical_act',
                                                  'sleep',
                                                  'social_act',
                                                  'food']

        xgboost.plot_importance(classifier.get_booster(), importance_type='weight')
        pyplot.savefig('depr(weight).png', bbox_inches='tight')


def train_test_general_model(filename):
    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, 2:-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)

    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

    classifier.get_booster().feature_names = ['mood',
                                              'physical_act',
                                              'sleep',
                                              'social_act',
                                              'food']

    print('Mean accuracy: ', accuracies.mean() * 100)
    print('Std of accuracy: ', accuracies.std() * 100)

    with open('depr_mode.pkl', 'wb+') as f:
        pickle.dump(classifier, f)
    xgboost.plot_importance(classifier.get_booster(), importance_type='total_gain', show_values=False)

    pyplot.savefig('depr(total_gain).png', bbox_inches='tight')
