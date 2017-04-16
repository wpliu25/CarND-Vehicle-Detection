import glob
import pickle
import os
from helper_functions import *

def readData(pickle_file = 'data_images_features.p'):

    # files
    cars = glob.glob('./vehicles/GTI_Far/*.png')
    cars += glob.glob('./vehicles/GTI_MiddleClose/*.png')
    cars += glob.glob('./vehicles/GTI_Left/*.png')
    cars += glob.glob('./vehicles/GTI_Right/*.png')
    cars += glob.glob('./vehicles/KITTI_extracted/*.png')
    not_cars = glob.glob('./non-vehicles/Extras/*.png')
    not_cars += glob.glob('./non-vehicles/GTI/*.png')

    # get features (shuffled, separate into train/test and normalized features)
    features_train, features_test, labels_train, labels_test = norm_shuffle(cars, not_cars)

    # Save the data for easy access
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'cars': cars,
                    'not_cars': not_cars,
                    'features_train': features_train,
                    'features_test': features_test,
                    'labels_train': labels_train,
                    'labels_test': labels_test
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    print('Data cached in pickle file.')

    return cars, not_cars, features_train, features_test, labels_train, labels_test

def loadData(data_file = 'data_images_features.p'):
    with open(data_file, mode='rb') as f:
        data = pickle.load(f)
    cars = data['cars']
    not_cars = data['not_cars']
    features_train = data['features_train']
    features_test = data['features_test']
    labels_train = data['labels_train']
    labels_test = data['labels_test']
    return cars, not_cars, features_train, features_test, labels_train, labels_test

if __name__ == "__main__":
    f = data_file = 'data_images_features.p'
    if(os.path.exists(f)):
        cars, not_cars, features_train, features_test, labels_train, labels_test = loadData(f)
    else:
        cars, not_cars, features_train, features_test, labels_train, labels_test = readData(f)

    print('Number of samples in training set: ', len(features_train))
    print('Number of samples in  test set: ',len(features_test))
