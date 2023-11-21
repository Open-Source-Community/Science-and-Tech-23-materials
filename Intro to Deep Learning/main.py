import numpy as np
import pandas as pd
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn import svm


print("loading data ...")
Train_File = os.listdir("E:\\PycharmProjects\\Task3\dogs-vs-cats\\train\\train2")

hog_images = []
features = []
training_cats = []
training_dogs = []


for image in Train_File:
    category = image.split('.')[0]
    if len(training_cats) < 1000 and category == 'cat':
        training_cats.append(0)
        img = imread("E:\\PycharmProjects\\Task3\dogs-vs-cats\\train\\train2\\" + image)
        img = resize(img, (128, 64))
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=True)
        features.append(fd)
        hog_images.append(hog_image)
    elif len(training_dogs) < 1000 and category == 'dog':
        training_dogs.append(1)
        img = imread("E:\\PycharmProjects\\Task3\dogs-vs-cats\\train\\train2\\" + image)
        img = resize(img, (128, 64))
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=True)
        features.append(fd)
        hog_images.append(hog_image)
    if len(training_cats) >= 1000 and len(training_dogs) >= 1000:
        break


LabelOfData = training_cats + training_dogs
print(len(LabelOfData))
#feature description + feature + category
df = pd.DataFrame({'hog Image': hog_images,'features': features, 'Category name (Cat or dog)': LabelOfData})
df = shuffle(df)

X = pd.DataFrame(df['features'].tolist())
Y = pd.DataFrame(df['Category name (Cat or dog)'].tolist())

print("Data Separated ...")

C = 0.1

svc = svm.SVC(kernel='linear', C=C).fit(X,Y)
linear_svc = svm.LinearSVC( C=C).fit(X,Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8 ,C=C).fit(X,Y)
poly_svc = svm.SVC(kernel='poly', degree=5, C= C).fit(X,Y)


Test_File = os.listdir("E:\\PycharmProjects\\Task3\dogs-vs-cats\\train\\train2\\")

hog_test_images = []
test_features = []
LabelOfTestData = []
testing_cats = []
testing_dogs = []

for image in Train_File:
    category = image.split('.')[0]
    number = image.split('.')[1]
    if len(testing_cats) < 100 and category == 'cat' and int(number) > 1000:
        testing_cats.append(0)
        img = imread("E:\\PycharmProjects\\Task3\dogs-vs-cats\\train\\train2\\" + image)
        img = resize(img, (128, 64))
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=True)
        test_features.append(fd)
        hog_test_images.append(hog_image)
    elif len(testing_dogs) < 100 and category == 'dog'and int(number) > 1000:
        testing_dogs.append(1)
        img = imread("E:\\PycharmProjects\\Task3\dogs-vs-cats\\train\\train2\\" + image)
        img = resize(img, (128, 64))
        #histogram of oriented gradients [Feature Descriptor]
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=True)
        test_features.append(fd)
        hog_test_images.append(hog_image)
    if len(testing_cats) >= 1000 and len(testing_dogs) >= 1000:
        break

LabelOfTestData = testing_cats + testing_dogs
print("LabelOfTestData length = ", len(LabelOfTestData))
df_test = pd.DataFrame({'hog Image': hog_test_images,'features': test_features, 'Category name (Cat or dog)': LabelOfTestData})
df_test = shuffle(df_test)

X_test = pd.DataFrame(df_test['features'].tolist())
Y_test = pd.DataFrame(df_test['Category name (Cat or dog)'].tolist())
#print(Y_test)
for i, clf in enumerate((svc, linear_svc, rbf_svc, poly_svc)):
    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == list(Y_test))
    print("Accuracy    =  ", accuracy)
