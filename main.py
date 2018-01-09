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

CSV_FILE_PATH = "/sumsung2T/udacity/steering_angle_data/driving_log.csv"
IMAGE_FILE_PATH = "/sumsung2T/udacity/steering_angle_data"
BATCH_SIZE = 32
EPOCH = 5
CORRECTION = 0.2
AUGMENTATION = True
IMAGE_SHAPE = [160,320,3]

logs = []
with open(CSV_FILE_PATH) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)

    for line in reader:

        if AUGMENTATION:

            log_center = [line[0], float(line[3])]
            log_left = [line[1].split()[0], float(line[3]) + CORRECTION]
            log_right = [line[2].split()[0], float(line[3]) - CORRECTION]

            flipped_image_file_name = 'IMG/flipped' + line[0].split('center')[1]
            flipped_image_full_file_name = IMAGE_FILE_PATH + '/' + flipped_image_file_name
            if not os.path.isfile(flipped_image_full_file_name):
                flipped_image = np.fliplr(cv2.imread(IMAGE_FILE_PATH + '/' + line[0]))
                cv2.imwrite(flipped_image_full_file_name, flipped_image)
            log_flipped = [flipped_image_file_name, -float(line[3])]
            logs.extend([log_center, log_left, log_right, log_flipped])

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
                name = IMAGE_FILE_PATH + '/' + batch_sample[0]
                image = cv2.imread(name)
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
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=IMAGE_SHAPE))
# first set of CONV => RELU => POOL
model.add(Convolution2D(6, 5, 5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second set of CONV => RELU => POOL
model.add(Convolution2D(16, 5, 5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# set of FC => RELU layers
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.7))

model.add(Dense(64))
model.add(Activation("relu"))


model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=len(training_logs)/BATCH_SIZE,
                    epochs=EPOCH,
                    validation_data=test_generator,
                    validation_steps=len(testing_logs)/BATCH_SIZE
                    )

