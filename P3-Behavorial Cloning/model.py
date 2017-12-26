# learn.py

from preprocessing import *
from neural_net import *
from json import dump, load
import pickle
from keras.models import model_from_json
from keras.optimizers import Adam
from datetime import datetime


def main(first_entry=False, correction=None, lr=1e-3):

    # read driving_log into dataframe 
    df = load_data()

    # get time
    df = capture_time(df) # added a time variable to df
    print(df.shape)

    # corrections

    if first_entry:
        # create left and right correction lists
        lr_corrections, rl_corrections, time_splits = [],[], []
        pickle.dump(lr_corrections, open('lr_corrections.p', 'wb'))
        pickle.dump(rl_corrections, open('rl_corrections.p', 'wb'))
    else:
        # load time splits to get last training date
        time_splits = pickle.load(open('time_splits.p', 'rb'))

        # upload left and right correction lists
        lr_corrections = pickle.load(open('lr_corrections.p', 'rb'))
        rl_corrections = pickle.load(open('rl_corrections.p', 'rb'))

        # register start and end period for a left correction
        if correction == 'left-right':
            duration = df.ix[df['time'] > time_splits[-1], 'time']
            lr_corrections.append((duration.min(), duration.max()))
            print(lr_corrections)
            # save
            pickle.dump(lr_corrections, open('lr_corrections.p', 'wb'))

        # register start and end period for a right correction
        if correction == 'right-left':
            duration = df.ix[df['time'] > time_splits[-1], 'time']
            rl_corrections.append((duration.min(), duration.max())) 
            # save
            pickle.dump(rl_corrections, open('rl_corrections.p', 'wb'))

    # save last_date
    time_splits.append(df['time'].max())
    pickle.dump(time_splits, open('time_splits.p', 'wb'))

    # fix left to right corrections (positives) - filter positive angles only
    for lt, ut in lr_corrections:
        df = df[(df['time'] < lt) | (df['time'] > ut) | (df['Steering Angle']>0)]  
        print(df.shape)

    # fix right to left corrections (negatives) - filter negative angles only
    for lt, ut in rl_corrections:
        df = df[(df['time'] < lt) | (df['time'] > ut) | (df['Steering Angle']<0)]   
        print(df.shape)

    # load images - at the time of train only, don't need to be saved with the dataframe
    X, y = load_images(df)

    # load neural network
    if not first_entry:
        with open('model.json', 'r') as f:
            model =  model_from_json(load(f))
        optimizer = Adam(lr=lr)
        model.compile(loss='mse', optimizer=optimizer)
        model.load_weights('model.h5')
    else:
        model = baseline_model()

    # train neural network
    model.fit(X, y, batch_size=20, nb_epoch=10)

    # save neural and network and last training date
    with open('model.json', 'w') as f:
        dump(model.to_json(), f)
    model.save_weights('model.h5')

    # force garbage collection
    import gc; gc.collect()

if __name__ == '__main__':
    # main(first_entry=True)
    main(lr=1e-4)
    # main(correction='left-right')
    # main(correction='right-left')

