import glob, os
import matplotlib.image as mpimg
from lesson_functions import *
from helper_functions import *

if __name__ == '__main__':
    test_images_path = os.path.join('test_images')
    test_images = sorted(glob.glob(os.path.join(test_images_path, 'test' + '*.jpg')))

    if os.path.exists(test_images_path):
        for idx, fname in enumerate(test_images):
            img = mpimg.imread(fname)
            windows = slide_window_scaled(img)
            windows_img = draw_boxes(img, windows, color=(0, 0, 255), thick=6)

            font_size = 30
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(img)
            ax1.set_title('orig', fontsize=font_size)
            ax2.imshow(windows_img)
            ax2.set_title('windows', fontsize=font_size)
            plt.rc('xtick', labelsize=font_size)
            plt.rc('ytick', labelsize=font_size)
            plt.show()
            #plt.savefig('./images/car_notcar.png')