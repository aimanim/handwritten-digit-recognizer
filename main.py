import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.src.saving import load_model
from tensorflow.keras import layers, models
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from customtkinter import *
import Image, ImageDraw
import pickle
from math import *
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

labels_list = [0,1,2,3,4,5,6,7,8,9]


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
        label.configure(text=f"Predicted Digit: {digit_label}", font=('Franklin Gothic Medium', 18, 'bold'))

    def clear():
        # Clear the SEEN canvas
        canvas.delete(ALL)
        # Clear the UNSEEN canvas
        w = image1.width
        h = image1.height
        draw.rectangle([0, 0, w, h], fill="black", width=0)
        # clear the label
        label.configure(text="")

    app = CTk()
    app.title("draw digit")
    frame = CTkFrame(app)

    canvas = CTkCanvas(app, width=280, height=280, bg="black")
    canvas.pack(padx=30, pady=30, anchor='nw', fill='both', expand=1)
    frame.pack()

    # PIL create an empty image and draw object to draw on.
    image1 = Image.new("RGB", (width, height), black)
    draw = ImageDraw.Draw(image1)

    canvas.bind("<Button-1>", get_x_and_y)
    canvas.bind("<B1-Motion>", draw_smth)

    button = CTkButton(frame, text="predict", font=('Franklin Gothic Medium', 15, 'bold'), command=predict)
    button.pack(pady=10)
    button = CTkButton(frame, text="clear", font=('Franklin Gothic Medium', 15, 'bold'), command=clear)
    button.pack(pady=10)

    label = CTkLabel(app, text="")
    label.pack(pady=25)

    app.mainloop()


def testing():

    def get_input():
        lbl2.configure(text=f"Testing accuracy: {round(evaluation[1],5)}", font=('Franklin Gothic Medium', 15, 'bold'))
        lbl3.configure(text=f"Testing loss: {round(evaluation[0],5)} ", font=('Franklin Gothic Medium', 15, 'bold'))
        start = int(entry1.get())
        end = int(entry2.get())
        plt.figure(figsize=(10, 5))
        for i in range(end - start + 1):
            plt.subplot(ceil((end - start + 1)/3), 3, i + 1)  # 3 columns
            plt.imshow(X_test[i + start - 1].reshape(28, 28), cmap='gray')  # Reshape to 28x28 and plot in grayscale
            plt.title(f"Predicted Label: {y_predicted_labels[i + start - 1]}")  # Add label as title
            plt.axis('off')  # Turn off axis
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    t = CTk()
    t.title("Test dataset")

    # Create a label
    lbl=CTkLabel(t, text="")
    lbl.pack(pady=3)
    label1 = CTkLabel(t, text="Choose between 1 and 10,000", font=('Franklin Gothic Medium', 15))
    label1.pack()
    label2 = CTkLabel(t, text="Enter starting image number:", font=('Franklin Gothic Medium', 16))
    label2.pack(padx=30,pady=20)

    # Create an entry widget for text input
    entry1 = CTkEntry(t, width=150)
    entry1.pack(padx=20)

    label3 = CTkLabel(t, text="Enter ending image number:", font=('Franklin Gothic Medium', 16))
    label3.pack(pady=20)
    entry2 = CTkEntry(t, width=150)
    entry2.pack(padx=20)

    button = CTkButton(t, text="Submit", font=('Franklin Gothic Medium', 18, 'bold'), command=get_input)
    button.pack(padx=10,pady=20)
    lbl2=CTkLabel(t,text="")
    lbl2.pack(pady=3)
    lbl3 = CTkLabel(t, text="")
    lbl3.pack(pady=10)

    t.mainloop()

def stats():
    t = CTk()
    t.title("Testing statistics")
    lbl1 = CTkLabel(t, text=f"Classification Report: ", font=('Franklin Gothic Medium', 15, 'bold'))
    lbl1.pack(pady=10, padx=20)
    lbl3 = CTkLabel(t, text=f"{classification_report(y_test, y_predicted_labels)} ", font=('Franklin Gothic Medium', 18))
    lbl3.pack(pady=10,padx=20)

    conf_matrix = confusion_matrix(y_test, y_predicted_labels)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(classification_report(y_test, y_predicted_labels))
    # Plot ROC curve for each class using one-vs-all approach
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(labels_list)):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_predicted_by_model[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'pink']
    for i, color in zip(range(len(labels_list)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Class Classification')
    plt.legend(loc="lower right")
    plt.show()

    # plot cnn confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix on Test Data')
    plt.xticks(ticks=np.arange(len(labels_list)), labels=labels_list, rotation=45)
    plt.yticks(ticks=np.arange(len(labels_list)), labels=labels_list, rotation=0)
    plt.show()
    t.mainloop()


class menu():
    def __init__(self):
        self.root = CTk()
        self.DisplayMenu()

    def CloseWindow(self):
        self.root.destroy()

    def DisplayMenu(self):
        self.root.geometry("600x600")
        self.root.title("Main Menu")
        self.root.protocol("WM_DELETE_WINDOW", self.CloseWindow)
        self.root.configure(bg="Black")
        title_lbl = CTkLabel(self.root, text="Handwritten Digit Recognizer", font=('Franklin Gothic Medium', 27, 'bold'))
        title_lbl.place(x=125, y=100)
        grp1 = CTkLabel(self.root, text="Aiman Imran 21K-4525", font=('Franklin Gothic Medium', 12))
        grp2 = CTkLabel(self.root, text="Bilal Hassan 21K-4669", font=('Franklin Gothic Medium', 12))
        grp1.place(x=440, y=500)
        grp2.place(x=440, y=520)
        g_btn = CTkButton(self.root, text="Training Graphs", font=('Century Gothic', 23), command=self.graphs)
        t_btn = CTkButton(self.root, text="Testing Stats", font=('Century Gothic', 23), command=self.test)
        ds_btn = CTkButton(self.root, text="Test on Dataset", font=('Century Gothic', 23), command=self.dataset)
        d_btn = CTkButton(self.root, text="Test on Drawing", font=('Century Gothic', 23), command=self.predict)
        g_btn.place(x=210, y=190)
        t_btn.place(x=230, y=260)
        ds_btn.place(x=212, y=330)
        d_btn.place(x=210, y=400)
        self.root.mainloop()

    def graphs(self):
        # Plot the accuracy and loss
        plt.figure(figsize=(10, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def predict(self):
        predict_digit()

    def dataset(self):
        testing()

    def test(self):
        stats()


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
        layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    learning_rate = 0.001
    batch_size = 64

    optimizer = Adam(learning_rate=learning_rate)
    convolutional_neural_network.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # Train the model
    history = convolutional_neural_network.fit(X_train, y_train, epochs=10, batch_size=batch_size)

    # Save the trained model and its history
    convolutional_neural_network.save('cnnmodel.keras')
    with open('training_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)
    print("Model saved.")
    history = history.history

evaluation = convolutional_neural_network.evaluate(X_test, y_test)
print("Test Accuracy:", evaluation[1])
y_predicted_by_model = convolutional_neural_network.predict(X_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted_by_model]
set_default_color_theme("green")
menu()
