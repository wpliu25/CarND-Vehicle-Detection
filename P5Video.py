# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from helper_functions import *
from lesson_functions import *
import sys
import matplotlib.pyplot as plt


def process_image(image):
    #result = AdvancedLaneLines(image, line, 0)
    windows = slide_window_scaled(image)

    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
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

