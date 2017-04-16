import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from lesson_functions import *
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import tree

def train_decision_tree(data, get_accuracy = False):
    clf = tree.DecisionTreeClassifier()
    # Check the training time for the decision tree
    t=time.time()
    clf.fit(data.features_train, data.labels_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train decision tree...')

    if(get_accuracy):
        # Check the score of the DT
        print('Test Accuracy of decision tree = ', round(clf.score(data.features_test, data.labels_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        pred = clf.predict(data.features_test)
        print('My decision tree predicts: ', pred[0:n_predict])
        print('For these',n_predict, 'labels: ', data.labels_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with decision tree')

        #acc = accuracy_score(pred, data.labels_test)
        #print("acc %f" % (round(acc, 3)))

    return clf

def train_SVM_LinearSVC(data, get_accuracy = False):

    # Use a linear SVC
    clf = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    clf.fit(data.features_train, data.labels_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVM Linear SVC...')

    if(get_accuracy):
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(clf.score(data.features_test, data.labels_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        pred = clf.predict(data.features_test)
        print('My SVC predicts: ', pred[0:n_predict])
        print('For these',n_predict, 'labels: ', data.labels_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

        #acc = accuracy_score(pred, data.labels_test)
        #print("acc %f" % (round(acc, 3)))

    return clf

def norm_shuffle(examples, not_examples, examples_features, not_examples_features):

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

        if(0):
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
            #plt.savefig('./output_images/norm_features.png')

        print('Number of training samples : ', len(X_train))
        print('Number of test samples : ', len(X_test))

        return X_train, X_test, y_train, y_test, X_scaler
    else:
        print('Your function only returns empty feature vectors...')

# not used in pipeline, exploratory only
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

def find_cars_scaled(image, clf, data, draw=False, ystart = 400, ystop = 656, scales = [1, 1.5, 2, 2.5], threshold=1):

    bbox_list_scaled = []
    for scale in scales:
        out_img, bbox_list = find_cars(image, ystart, ystop, scale, clf, data.X_scaler, data.orient, data.pix_per_cell,
                                       data.cell_per_block,
                                       data.spatial_size,
                                       data.hist_bins)
        bbox_list_scaled.extend(bbox_list)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, bbox_list_scaled)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    if(draw):
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()
        # plt.savefig('./output_images/find_cars.png')

        plt.imshow(out_img)
        fig = plt.figure(figsize=(12, 4))
        plt.title('find_cars')
        fig.tight_layout()
        plt.show()
        # plt.savefig('./output_images/find_cars.png')

    return draw_img