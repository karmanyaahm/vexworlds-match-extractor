#!/usr/bin/env python

# stdlib deps
from typing import Tuple
import argparse
import re
from os import system
from os.path import isfile, getsize

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
        action='append',
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


if True:
    success = 0
    lastmsg = 0
    class MyLogger:
        def __getattr__(self, name): 
            def f(msg):    # pretend this function is every function like debug, info, warning, error
                global lastmsg
                if "ETA" in msg: 
                    print("YoutubeDL:", msg, " " * (lastmsg - len(msg)), end = "\r")
                else: print("YoutubeDL:", msg)
                lastmsg = len(msg)
            
            return f

    def capture_filename(status):
        global args, success
        success+=1
        el = args.input_file.index(status['info_dict']['webpage_url']) 
        args.input_file[el] = status['filename']
        print(args.input_file)

    ydl_opts = {
        'logger': MyLogger(),
        'progress_hooks': [capture_filename]
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([i for i in args.input_file if i.startswith("http")])
    
    if not success:
        print("youtubedl didn't return a filename???")
        exit(0)
    
if not args.output_file:
    args.output_file = ''.join(args.input_file)

print(f'READING from "{args.input_file}"')
print(f'WRITING to "Match N - {args.output_file}"')

class CV2Reader:
    contents = []
    total = 0
    def __init__(self, names):
        for name in names:
            cap = cv2.VideoCapture(name)
            self.contents.append((name, cap, cap.get(cv2.CAP_PROP_FRAME_COUNT)) )
            self.total += cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(self.contents)

    def read(self, num):
        for name, cap, availableinthisone in self.contents:
            if num < availableinthisone:
                cap.set(cv2.CAP_PROP_POS_FRAMES, num)
                _, frame = cap.read()
                #print("ret", name, frame, num, availableinthisone)
                return name, frame, num
            else:
                num -= availableinthisone
        raise Exception("non existtant frame " + str(num))
        
    def close(self):
        for _, cap, _ in self.contents:
            cap.release()

            


# not needed on nixos at least
# Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/env tesseract'

# Read the video file
video_capture = CV2Reader(args.input_file)

ROIS = [(0, 0, 484 / 1920, 99 / 1080), [13 / 1920, 84 / 1080, 875 / 1920, 125 / 1080]]
REGEXES = [re.compile("QUAL ?(\d+)"), re.compile("QUALIFICATION (\d+)")]

FPS = 30
SKIP = 15 * FPS
print("Regions: ", ROIS)


frame_count = 0
last_frame_count = 0  # dV for prog bar only???

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
    # Read the frame
    _, frame, _ = video_capture.read(frame_num)
    num = get_num_from_frame(frame, which)
    return num


def search_for_match(m, which, lower_bound = 0):
    upper_bound = video_capture.total - 1
    while upper_bound - lower_bound >= 20 * FPS: # cutoff
        mid_frame = int((lower_bound + upper_bound) // 2)
        original_mid_frame = mid_frame
        if mid_frame == lower_bound:
            print("STUPID CANNOT ADVANCE ANYMORE")
            return None
        if DEBUG: print(lower_bound, mid_frame, upper_bound)

        num = None
        for n in range(200):
            if DEBUG: print("HI", mid_frame)
            # Read the frame
            _, frame, _ = video_capture.read(mid_frame)
            num = get_num_from_frame(frame, which)
            if not num:
                mid_frame = original_mid_frame + 10*SKIP * (n//2 + 1 if n%2 else -n//2)
            else:
                break

        print("FOUND match", num,"at frame", mid_frame, "/", video_capture.total)

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
    frame_num = search_for_match(m, 1, lower_bound=tupmatch[0])
    tupscore = lin_search_boundaries(frame_num, 1, 3)
    print(tupscore)
    matchst = video_capture.read(tupmatch[0])[2] / FPS
    matchend = video_capture.read(tupmatch[1])[2] / FPS
    matchinpname = video_capture.read(tupmatch[0])[0]

    scorest = video_capture.read(tupscore[0])[2] / FPS
    scoreend = video_capture.read(tupscore[1])[2] / FPS
    scoreinpname = video_capture.read(tupscore[0])[0]

    cmds = [
        f"ffmpeg -ss {matchst} -to {matchend} -i '{matchinpname}' -c copy 'tmpsegment1.mp4'",
        f"ffmpeg -ss {scorest} -to {scoreend} -i '{scoreinpname}' -c copy 'tmpsegment2.mp4'",
        f"echo \"file './tmpsegment1.mp4'\" > ./tmplist.txt",
        f"echo \"file './tmpsegment2.mp4'\" >> ./tmplist.txt",
        f"ffmpeg -f concat -safe 0 -i ./tmplist.txt -c copy 'Match {m} - {args.output_file}'",
    ]

    for i in cmds: print(i)
    if input("RUN [Y/N]: ").lower() == "y": 
        for i in cmds: system(i)

video_capture.close()