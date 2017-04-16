# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from helper_functions import *
from lesson_functions import *
import sys
import matplotlib.pyplot as plt
from process_data import *
from data_structure import *
from P5 import *

# setup data and classifier
clf, data = setup()

def process_image(image):
    out_img = find_cars_scaled(image, clf, data)
    return out_img

if __name__ == "__main__":

    if(len(sys.argv) < 3):
        print("Usage: ./P5Video.py <input_file> <output_file>\n")
        sys.exit(1)

    input = sys.argv[1]
    output = sys.argv[2]
    clip2 = VideoFileClip(input)
    output_clip = clip2.fl_image(process_image)
    output_clip.write_videofile(output, audio=False)

