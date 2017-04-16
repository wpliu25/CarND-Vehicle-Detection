import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from lesson_functions import *
from sklearn.model_selection import train_test_split

def norm_shuffle(examples, not_examples, examples_features, not_examples_features):

    # extract combined color and HOG features
    #examples_features = extract_features(examples, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9,
    #                    pix_per_cell=8, cell_per_block=2, hog_channel='ALL')

    #not_examples_features = extract_features(not_examples, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9,
    #                    pix_per_cell=8, cell_per_block=2, hog_channel='ALL')

    if len(examples_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((examples_features, not_examples_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(examples_features)), np.zeros(len(not_examples_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        # print('Using:', orient, 'orientations', pix_per_cell,
        #      'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        car_ind = np.random.randint(0, len(examples))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(examples[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.show()
        plt.savefig('./output_images/norm_features.png')

        print('Number of training samples : ', len(X_train))
        print('Number of test samples : ', len(X_test))

        return X_train, X_test, y_train, y_test, X_scaler
    else:
        print('Your function only returns empty feature vectors...')


def slide_window_scaled(image):
    all_windows = []

    x_start_stop_scaled = [[None, None], [None, None], [None, None], [None, None]]
    y_start_stop_scaled = [[int(image.shape[0] / 2), int(image.shape[0] / 2)+64*2], [int(image.shape[0] / 2)+32, int(image.shape[0] / 2)+64*4], [int(image.shape[0] / 2)+32, int(image.shape[0] / 2)+64*5], [int(image.shape[0] / 2)+32, None]]
    xy_window_scaled = [(64, 64), (128, 128), (200,200), (256, 256)]
    xy_overlap_scaled = [(0.5, 0.5), (0.6, 0.6), (0.6, 0.6), (0.6, 0.6)]

    for i in range(len(y_start_stop_scaled)):
        windows = slide_window(image, x_start_stop=x_start_stop_scaled[i], y_start_stop=y_start_stop_scaled[i],
                               xy_window=xy_window_scaled[i], xy_overlap=xy_overlap_scaled[i])

        all_windows += windows

    return all_windows