#!/usr/bin/env python

# stdlib deps
from typing import Tuple
import argparse
import re
from os import system

# need to install deps
import pytesseract
import cv2
from yt_dlp import YoutubeDL

# ARGS
def parse_arguments():
    def comma_separated_ints(value):
        try:
            return [int(x) for x in value.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError("Integers must be comma-separated")

    parser = argparse.ArgumentParser(description="Extract vids ig")
    parser.add_argument(
        "-i", "--input", dest="input_file", type=str, help="Path/vexworlds.tv URL to the input file",
        required = True,
    )
    parser.add_argument(
        "-o", "--output", dest="output_file", type=str, help="Path to the output file"
    )
    parser.add_argument(
        "-m",
        "--matches",
        dest="matches",
        type=comma_separated_ints,
        help="List of integers",
        required = True,
    )
    parser.add_argument("-d", dest="debug", action='store_true', help="Print more stuff")
    return parser.parse_args()


args = parse_arguments()
DEBUG = args.debug
print(args.matches)


if args.input_file.startswith("http"):
    success = 0
    class MyLogger:
        def __getattr__(self, name): 
            def f(msg):    # pretend this function is every function like debug, info, warning, error
                print("YoutubeDL:", msg)
            return f

    def capture_filename(status):
        global args, success
        success+=1
        args.input_file = status['filename']

    ydl_opts = {
        'logger': MyLogger(),
        'progress_hooks': [capture_filename]
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(["https://www.vexworlds.tv/#/viewer/broadcasts/qualification-matches-innovate-elk09hemvm4c9ewwwevn/o0pngqswkl9vdmx59rwp"])
    
    if not success:
        print("youtubedl didn't return a filename???")
        exit(0)
    
    


if not args.output_file:
    args.output_file = args.input_file


print(f'READING from "{args.input_file}"')
print(f'WRITING to "Match N - {args.output_file}"')

# not needed on nixos at least
# Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/env tesseract'

# Read the video file
video_capture = cv2.VideoCapture(args.input_file)

ROIS = [(0, 0, 484 / 1920, 99 / 1080), [13 / 1920, 84 / 1080, 875 / 1920, 125 / 1080]]
REGEXES = [re.compile("QUAL ?(\d+)"), re.compile("QUALIFICATION (\d+)")]

FPS = 30
SKIP = 15 * FPS
print("Regions: ", ROIS)


frame_count = 0
last_frame_count = 0  # dV for prog bar only???

total_frames_in = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

lastgroup = []

def get_num_from_frame(frame, which) -> int:
    x, y, w, h = ROIS[which]
    x = int(x * frame.shape[1])
    w = int(w * frame.shape[1])
    y = int(y * frame.shape[0])
    h = int(h * frame.shape[0])
    roi_frame = frame[y : y + h, x : x + w]

    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Perform OCR
    text = pytesseract.image_to_string(
        gray,
        lang="eng",
        config='--psm 7 -c tessedit_char_whitelist=" QUALIFICATION1234567890"',
    ).strip()
    if DEBUG: print(text)

    fs = REGEXES[which].findall(text)
    if len(fs) == 0:
        return None
    elif len(fs) == 1:
        if int(fs[0]) == 18:
            cv2.imshow("title", gray)
            cv2.waitKey()
        return int(fs[0])
    else:
        print("FOUND TOO MANY MATCHES?????", which, ROIS[which], text)
        cv2.imshow("title", frame)
        return None


def fetch_num_and_frame(frame_num, which):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    # Read the frame
    ret, frame = video_capture.read()
    num = get_num_from_frame(frame, which)
    return num


def search_for_match(m, which, lower_bound = 0):
    upper_bound = total_frames_in - 1
    while lower_bound <= upper_bound:
        mid_frame = int((lower_bound + upper_bound) // 2)
        if mid_frame == lower_bound:
            print("STUPID CANNOT ADVANCE ANYMORE")
            return None
        if DEBUG: print(lower_bound, mid_frame, upper_bound)

        num = None
        while True:
            if DEBUG: print("HI", mid_frame)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            # Read the frame
            ret, frame = video_capture.read()
            num = get_num_from_frame(frame, which)
            if not num:
                mid_frame += int(SKIP)
            else:
                break

        print("FOUND match", num,"at frame", mid_frame, "/", total_frames_in)

        # print(num, type(num), m, type(m))
        if num < m:
            lower_bound = mid_frame
        elif num > m:
            upper_bound = mid_frame
        else:
            return mid_frame


def lin_search_boundaries(frame_num, which, scroll_secs) -> Tuple[int, int]:
    num = fetch_num_and_frame(frame_num, which)
    print("Finding boundaries of clip around", frame_num, "for match", num)

    minframe = frame_num
    maxframe = frame_num
    while num == fetch_num_and_frame(minframe, which):
        print("Scrolling down to the start of the clip", (minframe, maxframe))
        minframe -= scroll_secs * FPS
    while num == fetch_num_and_frame(maxframe, which):
        print("Scrolling up to the end of the clip", (minframe, maxframe))
        maxframe += scroll_secs * FPS
    return (int(minframe - 0*FPS), int(maxframe + 0*FPS))




for m in args.matches:
    print("searching for m: match actual")
    frame_num = search_for_match(m, 0)
    tupmatch = lin_search_boundaries(frame_num, 0, 3)
    print("match: ", tupmatch)

    print("Searching for score")
    frame_num = search_for_match(m, 1)
    tupscore = lin_search_boundaries(frame_num, 1, 3)
    print(tupscore)
    cmds = [
        f"ffmpeg -ss {tupmatch[0] / FPS} -to {tupmatch[1] / FPS} -i '{args.input_file}' -c copy 'tmpsegment1.mp4'",
        f"ffmpeg -ss {tupscore[0] / FPS} -to {tupscore[1] / FPS} -i '{args.input_file}' -c copy 'tmpsegment2.mp4'",
        f"echo \"file './tmpsegment1.mp4'\" > ./tmplist.txt",
        f"echo \"file './tmpsegment2.mp4'\" >> ./tmplist.txt",
        f"ffmpeg -f concat -safe 0 -i ./tmplist.txt -c copy 'Match {m} - {args.output_file}'",
    ]

    for i in cmds: print(i)
    if input("RUN [Y/N]: ").lower() == "y": 
        for i in cmds: system(i)