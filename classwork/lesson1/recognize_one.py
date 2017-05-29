from sklearn import svm, metrics, datasets
import matplotlib.pyplot as plt
import random

from xgboost import XGBClassifier

digits = datasets.load_digits()

# image bytes - matrix
print(digits.images[0])

# 0 is value that corresponds
print(digits.target[0])

samples = len(digits.target)

print(samples)

# reshape -> flatting from matrix to vector, for comfortable using
data = digits.images.reshape(samples, -1)

print(data[0])

train_samples_size = int(samples * 0.5)
#
# classificator = svm.SVC()
classificator = svm.SVC(gamma=0.001)
#classificator = XGBClassifier()

classificator.fit(data[:train_samples_size], digits.target[:train_samples_size])

# testsing our studyied algo
expected = digits.target[train_samples_size:]
predicted = classificator.predict(data[train_samples_size:])

print("\nclassifier result:\n", metrics.classification_report(expected, predicted))

random_number = random.randint(0, train_samples_size - 10)

random_num_index = train_samples_size + random_number

image_prediction_target = list(zip(digits.images[random_num_index:],
                                   classificator.predict(data[random_num_index:]),
                                   digits.target[random_num_index:]))

for index, (image, prediction, target) in enumerate(image_prediction_target[:10]):
    plt.subplot(1, 10, index + 1)
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.title("Prediction: %i\nTarget: %i" % (prediction, target))

plt.show()
