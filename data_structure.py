flatten = lambda l: [item for sublist in l for item in sublist]

class dataStructure():
    def __init__(self, features_train, features_test, labels_train, labels_test, X_scaler, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, y_start_stop, hist_range):

        # data
        self.features_train = features_train
        self.features_test = features_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.X_scaler = X_scaler
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.y_start_stop = y_start_stop
        self.hist_range = hist_range

        # bbox
        self.running_average_n = 8
        self.running_average_index = 0
        self.bbox_list = []
        for i in range(self.running_average_n):
            self.bbox_list.append([])

    def insert_bbox_list(self, bboxes):
        self.bbox_list[self.running_average_index] = bboxes
        self.running_average_index = self.running_average_index + 1
        if(self.running_average_index >= self.running_average_n-1):
            self.running_average_index = 0

        return flatten(self.bbox_list)