import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Define data directories
train_dir = 'D:/Lung Cancer Detection Using CNN/JPDL13-Lung Cancer Detection Using CNN/SOURCE CODE/CT lung cancer/model/train'
test_dir = 'D:/Lung Cancer Detection Using CNN/JPDL13-Lung Cancer Detection Using CNN/SOURCE CODE/CT lung cancer/model/test'

# Define image dimensions and other parameters
img_height = 224
img_width = 224
batch_size = 32
epochs = 5

# Data generators
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(img_height, img_width),
    class_mode='categorical')

val_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
val_data_gen = val_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir,
    shuffle=True,
    target_size=(img_height, img_width),
    class_mode='categorical')

# Model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), input_shape=(img_height, img_width, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data_gen,
                    epochs=epochs,
                    validation_data=val_data_gen)

# Evaluate the model
y_true = val_data_gen.classes
y_pred = np.argmax(model.predict(val_data_gen), axis=-1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)


# Plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plt.figure(figsize=(8, 6))
plot_confusion_matrix(cm, classes=['Cancerous', 'Non-Cancerous'])
plt.show()

# Save the model
model.save('cancer.h5')
