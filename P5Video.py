# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from helper_functions import *
from lesson_functions import *
import sys
import matplotlib.pyplot as plt
from process_data import *
from data_structure import *
from P5 import *

# load data
f = './data_images_features.p'
if (os.path.exists(f)):
    cars, not_cars, examples_features, not_examples_features, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop, hist_range = loadData(
        f)
else:
    cars, not_cars, examples_features, not_examples_features, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop, hist_range = readData(
        f)

features_train, features_test, labels_train, labels_test, X_scaler = norm_shuffle(cars, not_cars, examples_features,
                                                                                  not_examples_features)

data = dataStructure(features_train, features_test, labels_train, labels_test, X_scaler, color_space, orient,
                     pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat,
                     hog_feat, y_start_stop, hist_range)

clf = train_SVM_LinearSVC(data, True)

def process_image(image):
    #result = AdvancedLaneLines(image, line, 0)
    #windows = slide_window_scaled(image)

    #window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    #return window_img


    windows = slide_window_scaled(image)
    hot_windows = search_windows(image, windows, clf, data.X_scaler, color_space=data.color_space,
                                 spatial_size=data.spatial_size, hist_bins=data.hist_bins,
                                 hist_range=data.hist_range,
                                 orient=data.orient, pix_per_cell=data.pix_per_cell,
                                 cell_per_block=data.cell_per_block,
                                 hog_channel=data.hog_channel, spatial_feat=data.spatial_feat,
                                 hist_feat=data.hist_feat, hog_feat=data.hog_feat)
    draw_image = np.copy(image)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    return window_img


if __name__ == "__main__":

    if(len(sys.argv) < 3):
        print("Usage: ./P5Video.py <input_file> <output_file>\n")
        sys.exit(1)

    input = sys.argv[1]
    output = sys.argv[2]
    clip2 = VideoFileClip(input)
    output_clip = clip2.fl_image(process_image)
    output_clip.write_videofile(output, audio=False)

