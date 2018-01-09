import csv
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

CSV_FILE_PATH = "/Users/prateekagrawal/Desktop/my_git/udacity/udacity_sim_data/data/driving_log.csv"
IMAGE_FILE_PATH = "/Users/prateekagrawal/Desktop/my_git/udacity/udacity_sim_data/data"

logs = []
with open(CSV_FILE_PATH) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        logs.append(line)

print("total log files found are: ", len(logs))

images = []
steering_angles = []

for log in logs[1:]:
    center_image_path = IMAGE_FILE_PATH + '/' + log[0]
    image = cv2.imread(center_image_path)
    images.append(image)
    steering_angles.append(float(log[3]))

X_train,  X_test, y_train, y_test = train_test_split(images, steering_angles, shuffle=True, test_size=0.2)


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

