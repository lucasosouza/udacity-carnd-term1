# preprocessing.py

import pandas as pd
import numpy as np
from datetime import datetime
import cv2
from PIL import Image

######################################## MAIN FUNCTIONS

def load_data():
    ## 1. load data
    columns = ['Center Image','Left Image','Right Image','Steering Angle','Throttle', 'Break', 'Speed']
    df = pd.read_csv('driving_log.csv', names=columns)
    return df

def capture_time(df):
    """ Capture time in image name """

    df['time'] = df['Center Image'].apply(to_time)
    return df


def load_images(df, camera_position='Center Image', model_name='center'):
    """ Load and preprocess images into the driving dataframe """

    # 1. load images
    c_load_image = lambda img:load_image(img, camera_position=camera_position)
    df = df.apply(c_load_image, axis=1)
    
    # 2. separate into X and y
    X = np.array(df[camera_position].values.tolist())
    y = df['Steering Angle']

    ## 3. improve images
    # neural net version
    X = np.array([process_image(img) for img in X]).reshape(-1,16,32,3)

    return X, y

def remove_extreme_angles(df):
    return df.ix[(df['Steering Angle'] < .95) & (df['Steering Angle'] > -.95), :]

def remove_zeros(df):
    return df.ix[df['Steering Angle'] != 0, :]

def smooth_angles(df):

    ## get time differences
    df['time'] = df['Center Image'].apply(to_time)
    df['time_diff'] = (df['time'] - df['time'].shift(1)).fillna(value=0).apply(seconds_in_timedelta)

    ## divide into groups
    df['group'] = 0
    i = 1
    for row in df.iterrows():
        if row[1]['time_diff'] > 1:
            i+=1
        df.ix[row[0], 'group'] = i

    ## group by  
    vc = df['group'].value_counts()

    # correction remaining
    for group, count in zip(vc.index, vc):
        if group not in vc.index[5:7]:
            # window size will change depending on group size, to a max of 30
            smooth_window_len=min(30,ceil(count/2))
            if smooth_window_len > 3:
                print(group, count, smooth_window_len)
                df.ix[df['group']==group, 'Smoothed Angle'] = \
                smooth(df.ix[df['group']==group, 'Steering Angle'], window_len=smooth_window_len)[:count]
            else:
                df.ix[df['group']==group, 'Smoothed Angle'] = df.ix[df['group']==group, 'Steering Angle']

    return df

######################################## SUPPORT FUNCTIONS

def to_time(s):
    """ Captures time from image name and convert to time variable """
    s = s.replace('/Users/lucasosouza/Documents/CarND/P3-final/IMG/center_', '')
    s = s.replace('.jpg', '')
    s = datetime.strptime(s, '%Y_%m_%d_%H_%M_%S_%f')
    
    return s

def seconds_in_timedelta(td):
    """ Calculate the number of seconds in a timedelta """
    return td.seconds


def center_scale(arr):
    """ Center and scale values """

    xmax, xmin = arr.max(), arr.min()
    arr = ((arr - xmin) / (xmax - xmin)) -.5
    return arr

def load_image(row, camera_position='Center Image'):
    """ Converts file name into an array that corresponds to the image pixels """

    base_path = '/Users/lucasosouza/Documents/CarND/P3/'
    image_path = row[camera_position].replace(base_path, '').strip()
    row[camera_position] = np.asarray(Image.open(image_path))
    return row

def process_image_nvidia(img, resize=True, yuv=True):
    """ Steps to enhance image prior to classification """

    # resize
    if resize:
        img = cv2.resize(img, (66,200))
    # convert to YUV space, isolate Y channel
    if yuv:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)    

    return img    

def process_image(img, resize=True, yuv=False, histeq=False, adapthisteq=False, edge=False):
    """ Steps to enhance image prior to classification """

    # resize
    if resize:
        img = cv2.resize(img, (16,32))
    # convert to YUV space, isolate Y channel
    if yuv:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
    # preprocess Y with global histogram equalization (histeq)
    if histeq:
        img = cv2.equalizeHist(img)
    # preprocess Y with local histogram equalization (adapthisteq)
    if adapthisteq:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        img = clahe.apply(img)
    # edge detection - substract blurred image from original image
    if edge:
        gaussian_filter = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=3)
        img = cv2.addWeighted(img, 1.5, gaussian_filter, -0.5, gamma=1)
    return img #.reshape(32, 32, 1)

# smooth function, adapted from scipy formula at http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat':
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

if __name__=='__main__':
    df = load_data()
    # df = remove_extreme_angles(df)
    df = smooth_angles(df)
    df.to_pickle('data.p')
