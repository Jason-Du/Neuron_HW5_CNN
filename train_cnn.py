from keras.datasets import mnist
import numpy as np
from cnn import CNN
import matplotlib.pyplot as plt
import pickle

np.set_printoptions(edgeitems=100, linewidth=200000)

cnn = CNN(6, 12)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = (train_images / 255) - 0.5

test_images = (test_images/ 255) - 0.5

stats = cnn.train(train_images[:1000], train_labels[:1000], test_images[:1000], test_labels[:1000], 25, 0.01)
epochs = stats[0]
avg_losses = stats[1]
accuracies = stats[2]

# with open("artifacts/model.bin", "wb") as f:
#   pickle.dump(cnn, f)

fig = plt.figure()
plt.subplots_adjust(hspace=0.5)

g1 = fig.add_subplot(2, 1, 1, ylabel="Loss", xlabel="Epoch")
g1.plot(epochs, avg_losses, label="Avg loss", color="red")
g1.legend(loc="center")

g2 = fig.add_subplot(2, 1, 2, ylabel="Accuracy", xlabel="Epoch")
g2.plot(epochs, accuracies, label="Accuracy", color="green")
g2.legend(loc="center")

plt.show()
