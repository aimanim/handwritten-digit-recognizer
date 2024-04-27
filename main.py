import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.src.saving import load_model
from tensorflow.keras import layers,models
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from tkinter import *
import Image, ImageDraw
import pickle

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
X_train = X_train / 255
X_test = X_test / 255
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Check if model has already been trained and saved
try:
    convolutional_neural_network = load_model('cnnmodel.keras')
    print("Model loaded successfully.")
    with open('training_history.pkl', 'rb') as file:
        history = pickle.load(file)
except:
    print("Model not found. Training a new model...")
    # Define the model architecture
    convolutional_neural_network = models.Sequential([
        layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    convolutional_neural_network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history=convolutional_neural_network.fit(X_train, y_train, epochs=10)

    # Save the trained model and its history
    convolutional_neural_network.save('cnnmodel.keras')
    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    print("Model saved.")

# Evaluate the model
evaluation = convolutional_neural_network.evaluate(X_test, y_test)
print("Test Accuracy:", evaluation[1])
y_predicted_by_model = convolutional_neural_network.predict(X_test)

print(np.argmax(y_predicted_by_model[0]))

y_predicted_labels = [np.argmax(i) for i in y_predicted_by_model]
print(y_predicted_labels[10:16])

plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i+1)  # 1 row, 5 columns, position i+1
    plt.imshow(X_test[i+10].reshape(28, 28), cmap='gray')  # Reshape to 28x28 and plot in grayscale
    plt.title(f"Label: {y_test[i+10]}")  # Add label as title
    plt.axis('off')  # Turn off axis
plt.tight_layout()  # Adjust layout for better spacing
plt.show()

def predict_digit():
    width = 280
    height = 280
    black = (0, 0, 0)
    image1 = None

    def get_x_and_y(event):
        global lasx, lasy
        lasx, lasy = event.x, event.y

    def draw_smth(event):
        global lasx, lasy
        x, y = event.x, event.y
        r = 10  # radius of the circle
        canvas.create_oval(x - r, y - r, x + r, y + r, outline='#FFFFFF', fill='#FFFFFF')
        draw.ellipse([(x - r, y - r), (x + r, y + r)], outline='#FFFFFF', fill='#FFFFFF')

    def predict():
        image_gray = image1.convert("L")
        img = image_gray.resize((28, 28))
        img_array = np.array(img)

        # Normalize pixel values
        img_array = img_array / 255.0

        # Reshape array to match model input shape
        img_array = img_array.reshape(1, 28, 28, 1)

        # Use your trained model to predict the label for the image
        prediction = convolutional_neural_network.predict(img_array)
        digit_label = np.argmax(prediction)
        label.config(text=f"Predicted Digit: {digit_label}", font=('Franklin Gothic Medium', 15,'bold'))

    def clear():
        # Clear the SEEN canvas
        canvas.delete(ALL)
        # Clear the UNSEEN canvas
        w = image1.width
        h = image1.height
        draw.rectangle([0, 0, w, h], fill="black", width=0)
        #clear the label
        label.config(text="")

    def show():
        print("'show' button pressed")
        image1.show()

    # root is the main window used in our GUI, it builds off of tkinter.
    root = Tk()

    canvas = Canvas(root, width=280, height=280, bg="black")
    canvas.pack(anchor='nw', fill='both', expand=1)

    # PIL create an empty image and draw object to draw on.
    image1 = Image.new("RGB", (width, height), black)
    draw = ImageDraw.Draw(image1)

    canvas.bind("<Button-1>", get_x_and_y)
    canvas.bind("<B1-Motion>", draw_smth)

    button = Button(text="predict", font=('Franklin Gothic Medium', 12,'bold'), command=predict, width=15, height=1,
                    bd=2,   #border width
                    relief='raised',  # button relief style
                    bg='red4',     # button background color
                    fg='white')    # button foreground (text) color
    button.pack(pady=5)
    button = Button(text="clear", font=('Franklin Gothic Medium', 12,'bold'), command=clear, width=15, height=1,
                    bd=2, relief='raised', bg='red4',  fg='white')
    button.pack(pady=5)
    button = Button(text="show", font=('Franklin Gothic Medium', 12,'bold'), command=show, width=15, height=1,
                    bd=2, relief='raised', bg='red4',  fg='white')
    button.pack(pady=5)

    label = Label(root, text="")
    label.pack(pady=10)

    root.mainloop()
# Plot the accuracy and loss
plt.figure(figsize=(10, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.tight_layout()
plt.show()
predict_digit()
