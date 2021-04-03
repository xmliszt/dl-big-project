import os
import matplotlib
import pylab
import librosa
import librosa.display
import numpy as np
from shutil import copytree, ignore_patterns
import re

# No pictures displayed 
matplotlib.use('Agg') 

inputpath = os.path.join(os.getcwd(), 'dataset', 'genres_original')
outputpath = os.path.join(os.getcwd(), 'dataset', 'images_original')

# Copy tree structure
try:
    copytree(inputpath, outputpath, ignore=ignore_patterns('*.wav'))
except OSError as e:
    pass

for dirpath, dirnames, filenames in os.walk(inputpath):

    if filenames == []:
        continue

    print("Now converting: ", dirpath)
    for file in filenames:
        inputfile = os.path.join(dirpath, file)
        sig, fs = librosa.load(inputfile) 
        outputfile = re.sub('genres', 'images', inputfile).replace('.wav','.jpg')

        pylab.axis('off') # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
        S = librosa.feature.melspectrogram(y=sig, sr=fs)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        pylab.savefig(outputfile, bbox_inches=None, pad_inches=0)
        pylab.close()
    print("Done converting:", dirpath)