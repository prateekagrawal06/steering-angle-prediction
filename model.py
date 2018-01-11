import csv
import cv2
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import os
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.layers import Cropping2D
from keras.callbacks import ModelCheckpoint


CSV_FILE_PATH = "/sumsung2T/udacity/steering_angle_data/track_1/driving_log.csv"
IMAGE_FILE_PATH = "/sumsung2T/udacity/steering_angle_data/track_1/IMG"

CSV_FILE_PATH_2 = "/sumsung2T/udacity/steering_angle_data/track_2/driving_log.csv"
IMAGE_FILE_PATH_2 = "/sumsung2T/udacity/steering_angle_data/track_2/IMG"

BATCH_SIZE = 128
EPOCH = 50
CORRECTION = 0.2
AUGMENTATION = True
IMAGE_SHAPE = [160, 320, 3]
DROPOUT = 0.7

logs = []
with open(CSV_FILE_PATH) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)

    for line in reader:

        log_center = [line[0], float(line[3])]
        log_left = [line[1].split()[0], float(line[3]) + CORRECTION]
        log_right = [line[2].split()[0], float(line[3]) - CORRECTION]

        flipped_image_file_name = 'flipped' + line[0].split('center')[1]
        flipped_image_full_file_name = IMAGE_FILE_PATH + '/' + flipped_image_file_name

        if not os.path.isfile(flipped_image_full_file_name):
            flipped_image = np.fliplr(cv2.imread(line[0]))
            cv2.imwrite(flipped_image_full_file_name, flipped_image)
        log_flipped = [flipped_image_full_file_name, -float(line[3])]

        if AUGMENTATION:
            logs.extend([log_center, log_left, log_right, log_flipped])
        else:
            logs.append(log_center)


with open(CSV_FILE_PATH_2) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)

    for line in reader:

        log_center = [line[0], float(line[3])]
        log_left = [line[1].split()[0], float(line[3]) + CORRECTION]
        log_right = [line[2].split()[0], float(line[3]) - CORRECTION]

        flipped_image_file_name = 'flipped' + line[0].split('center')[1]
        flipped_image_full_file_name = IMAGE_FILE_PATH_2 + '/' + flipped_image_file_name

        if not os.path.isfile(flipped_image_full_file_name):
            flipped_image = np.fliplr(cv2.imread(line[0]))
            cv2.imwrite(flipped_image_full_file_name, flipped_image)
        log_flipped = [flipped_image_full_file_name, -float(line[3])]

        if AUGMENTATION:
            logs.extend([log_center, log_left, log_right, log_flipped])
        else:
            logs.append(log_center)

print("total log files found are: ", len(logs))
training_logs, testing_logs = train_test_split(logs, test_size=0.2, random_state=1234, shuffle=True)
print("training logs size {}".format(len(training_logs)))
print("testing logs size {}".format(len(testing_logs)))


def batch_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                steering_angle = float(batch_sample[1])
                images.append(image)
                angles.append(steering_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = batch_generator(training_logs, BATCH_SIZE)
test_generator = batch_generator(testing_logs, BATCH_SIZE)

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=IMAGE_SHAPE))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# conv --> relu -- > conv --> relu --> pooling
model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv --> relu -- > conv --> relu --> pooling
model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

# conv --> relu -- > conv --> relu --> pooling
model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# conv --> relu -- > conv --> relu --> pooling
model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# set of FC => RELU layers
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(1))
print(model.summary())

adam = Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam)
checkpoint = ModelCheckpoint('model_track_1.h5', verbose=2, period=1)
model.fit_generator(train_generator,
                    steps_per_epoch=len(training_logs)/BATCH_SIZE,
                    epochs=EPOCH,
                    callbacks=[checkpoint],
                    validation_data=test_generator,
                    validation_steps=len(testing_logs)/BATCH_SIZE
                    )

# model.save('model_track_1.h5')

