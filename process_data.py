import glob
import pickle
import os
from helper_functions import *
from data_structure import *

### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
hist_range=(0, 256)

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
    # extract combined color and HOG features
    examples_features = extract_features(cars, cspace=color_space, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

    not_examples_features = extract_features(not_cars, cspace=color_space, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

    # Save the data for easy access
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'cars': cars,
                    'not_cars': not_cars,
                    'examples_features': examples_features,
                    'not_examples_features': not_examples_features,
                    'color_space': color_space,
                    'orient': orient,
                    'pix_per_cell': pix_per_cell,
                    'cell_per_block': cell_per_block,
                    'hog_channel': hog_channel,
                    'spatial_size': spatial_size,
                    'hist_bins': hist_bins,
                    'spatial_feat': spatial_feat,
                    'hist_feat': hist_feat,
                    'hog_feat': hog_feat,
                    'y_start_stop': y_start_stop,
                    'hist_range': hist_range
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    print('Data cached in pickle file.')

    return cars, not_cars, examples_features, not_examples_features, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop, hist_range

def loadData(data_file = 'data_images_features.p'):
    with open(data_file, mode='rb') as f:
        data = pickle.load(f)
    cars = data['cars']
    not_cars = data['not_cars']
    examples_features = data['examples_features']
    not_examples_features = data['not_examples_features']
    color_space = data['color_space']
    orient = data['orient']
    pix_per_cell = data['pix_per_cell']
    cell_per_block = data['cell_per_block']
    hog_channel = data['hog_channel']
    spatial_size = data['spatial_size']
    hist_bins = data['hist_bins']
    spatial_feat = data['spatial_feat']
    hist_feat = data['hist_feat']
    hog_feat = data['hog_feat']
    y_start_stop = data['y_start_stop']
    hist_range = data['hist_range']

    return cars, not_cars, examples_features, not_examples_features, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop, hist_range

def setup(f = './data_images_features.p'):
    # load data
    if (os.path.exists(f)):
        cars, not_cars, examples_features, not_examples_features, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop, hist_range = loadData(f)
    else:
        cars, not_cars, examples_features, not_examples_features, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop, hist_range = readData(f)

    features_train, features_test, labels_train, labels_test, X_scaler = norm_shuffle(cars, not_cars, examples_features, not_examples_features)

    data = dataStructure(features_train, features_test, labels_train, labels_test, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop, hist_range)

    # train
    # color, spatial: Test Accuracy of SVC = 0.9181, 0.01135 Seconds to predict 10 labels with SVC
    # color, spatial, hog0: Test Accuracy of SVC = 0.9716, 0.01754 Seconds to predict 10 labels with SVC
    # color, spatial, hogAll: Test Accuracy of SVC = 0.9797, 0.0286 Seconds to predict 10 labels with SVC
    clf = train_SVM_LinearSVC(data, True)

    # color, spatial: Test Accuracy of DT = 0.9077, 0.01928 Seconds to predict 10 labels with DT
    # color, spatial, hog0: Test Accuracy of DT = 0.9223, 0.03695 Seconds to predict 10 labels with DT
    # color, spatial, hogAll: Test Accuracy of DT = 0.92, 0.05914 Seconds to predict 10 labels with DT
    #clf = train_decision_tree(data, True)

    return clf, data

if __name__ == "__main__":
    f = data_file = 'data_images_features.p'
    if(os.path.exists(f)):
        cars, not_cars, examples_features, not_examples_features, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop, hist_range = loadData(f)
    else:
        cars, not_cars, examples_features, not_examples_features, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop, hist_range = readData(f)

    print('Number of samples of cars: ', len(examples_features))
    print('Number of samples of not cars: ',len(not_examples_features))
