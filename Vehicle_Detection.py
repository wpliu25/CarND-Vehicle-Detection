import glob, os
import matplotlib.image as mpimg
from lesson_functions import *
from helper_functions import *
from P5 import *

if __name__ == '__main__':
    test_images_path = os.path.join('test_images')
    test_images = sorted(glob.glob(os.path.join(test_images_path, 'test' + '*.jpg')))

    # load data
    f = './data_images_features.p'
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

    if os.path.exists(test_images_path):
        for idx, fname in enumerate(test_images):
            image = mpimg.imread(fname)
            draw_image = np.copy(image)

            if(0):
                windows = slide_window_scaled(image)
                windows_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)

                font_size = 30
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                ax1.imshow(image)
                ax1.set_title('orig', fontsize=font_size)
                ax2.imshow(windows_img)
                ax2.set_title('windows', fontsize=font_size)
                plt.rc('xtick', labelsize=font_size)
                plt.rc('ytick', labelsize=font_size)
                plt.show()
                #plt.savefig('./images/car_notcar.png')

            # Uncomment the following line if you extracted training
            # data from .png images (scaled 0 to 1 by mpimg) and the
            # image you are searching is a .jpg (scaled 0 to 255)
            image = image.astype(np.float32)/255

           # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
           #                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))

            windows = slide_window_scaled(image)
            hot_windows = search_windows(image, windows, clf, data.X_scaler, color_space=data.color_space,
                                         spatial_size = data.spatial_size, hist_bins = data.hist_bins,
                                         hist_range = data.hist_range,
                                         orient = data.orient, pix_per_cell = data.pix_per_cell,
                                         cell_per_block = data.cell_per_block,
                                         hog_channel = data.hog_channel, spatial_feat = data.spatial_feat,
                                         hist_feat = data.hist_feat, hog_feat = data.hog_feat)

            window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

            plt.imshow(window_img)