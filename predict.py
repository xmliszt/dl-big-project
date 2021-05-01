'''To be used by Backend server to run prediction'''

import os
import sys
from evaluate import predict

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

audio_filename = sys.argv[1]

top5 = predict(os.path.join(
    DIR_PATH, "model3.pth"), os.path.join(
    DIR_PATH, "backend", "upload", audio_filename))

print(top5)
