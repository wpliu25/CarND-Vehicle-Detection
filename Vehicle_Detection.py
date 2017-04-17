import glob, os
import matplotlib.image as mpimg
from lesson_functions import *
from helper_functions import *
from process_data import *

if __name__ == '__main__':
    test_images_path = os.path.join('test_images')
    test_images = sorted(glob.glob(os.path.join(test_images_path, 'out' + '*.jpg')))

    clf, data = setup()

    if os.path.exists(test_images_path):
        for idx, fname in enumerate(test_images):
            image = mpimg.imread(fname)
            draw_image = np.copy(image)

            # Uncomment the following line if you extracted training
            # data from .png images (scaled 0 to 1 by mpimg) and the
            # image you are searching is a .jpg (scaled 0 to 255)
            #image = image.astype(np.float32)/255

            draw_img = find_cars_scaled(image,clf,data, draw=True, threshold=3, index=idx)


