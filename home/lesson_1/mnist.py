from sklearn import svm, metrics, datasets
import matplotlib.pyplot as plt
import random
from xgboost import XGBClassifier

digits = datasets.load_digits()
print(digits.images[0])
print(digits.target[0])
samples = len(digits.target)
print(samples)
data = digits.images.reshape(samples, -1)

print(data[0])

train_samples = int(samples * 0.5)

# classificator = svm.SVC(gamma=0.001)
classificator = XGBClassifier()

classificator.fit(data[:train_samples], digits.target[:train_samples])

expected = digits.target[train_samples:]
predicted = classificator.predict(data[train_samples:])

print("Classifier result:\n", metrics.classification_report(expected, predicted))

random_number = random.randint(0, train_samples - 10)
number = train_samples + random_number
image_and_prediction_and_target = list(zip(digits.images[number:],
                                           classificator.predict(data[number:]),
                                           digits.target[number:]))

for index, (image, predicted, target) in enumerate(image_and_prediction_and_target[:10]):
    plt.subplot(1, 10, index + 1)
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.title("Prediction: %s\nTarget: %s" % (str(predicted), str(target)))
plt.show()
