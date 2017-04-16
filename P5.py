import time
from process_data import *
from helper_functions import *
from lesson_functions import *
from data_structure import dataStructure
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#from class_vis import prettyPicture, output_image
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

if __name__ == "__main__":

    # load data
    f = './data_images_features.p'
    if(os.path.exists(f)):
        cars, not_cars, features_train, features_test, labels_train, labels_test = loadData(f)

    data = dataStructure(features_train, features_test, labels_train, labels_test)

    # train
    # color, spatial: Test Accuracy of SVC = 0.9181, 0.01135 Seconds to predict 10 labels with SVC
    # color, spatial, hog0: Test Accuracy of SVC = 0.9716, 0.01754 Seconds to predict 10 labels with SVC
    # color, spatial, hogAll: Test Accuracy of SVC = 0.9797, 0.0286 Seconds to predict 10 labels with SVC
    clf = train_SVM_LinearSVC(data, True)

    # color, spatial: Test Accuracy of DT = 0.9077, 0.01928 Seconds to predict 10 labels with DT
    # color, spatial, hog0: Test Accuracy of DT = 0.9223, 0.03695 Seconds to predict 10 labels with DT
    # color, spatial, hogAll: Test Accuracy of DT = 0.92, 0.05914 Seconds to predict 10 labels with DT
    #clf = train_decision_tree(data, True)

    #prettyPicture(clf, data.features_test, data.labels_test)
    #output_image("test.png", "png", open("test.png", "rb").read())


