import concurrent.futures
import os
import time

from tools import create_filenames

USER_ID = 89
directory = 'data_for_fe'
data_sources = [1, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 71]



def extract_features(user_directory):
    print(f'Feature extraction started for {user_directory}')



    return f'Feature extraction finished for {user_directory}'


def main():
    create_filenames(USER_ID, data_sources)

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
