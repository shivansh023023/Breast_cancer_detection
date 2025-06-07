# Breast Cancer Prediction using Neural Networks

This project demonstrates how to build and train a simple neural network model to predict whether a breast tumor is malignant or benign based on features extracted from cell nuclei.

## Dataset

The dataset used in this project is the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The target variable is whether the tumor is Malignant (0) or Benign (1).

You can access the dataset used in this specific Colab notebook via this link:
[Breast Cancer Dataset](https://drive.google.com/file/d/1HQqUCF23fihnE3aCbx2MNPmER9o_If_m/view)

## Project Structure

The core of the project is a Google Colab notebook that performs the following steps:

1.  **Data Loading and Exploration:**
    *   Loads the breast cancer dataset from scikit-learn.
    *   Converts the dataset into a pandas DataFrame for easier manipulation.
    *   Explores the data by checking its shape, information, missing values, descriptive statistics, and the distribution of the target variable.
2.  **Data Preprocessing:**
    *   Separates the features (X) and the target variable (Y).
    *   Splits the data into training and testing sets.
    *   Standardizes the features using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1. This is important for many machine learning algorithms, including neural networks.
3.  **Model Building:**
    *   Creates a simple Sequential neural network model using TensorFlow/Keras.
    *   The model consists of a Flatten layer, a Dense hidden layer with ReLU activation, and an output Dense layer with Sigmoid activation for binary classification.
4.  **Model Compilation and Training:**
    *   Compiles the model with the Adam optimizer, sparse categorical crossentropy loss, and accuracy as the evaluation metric.
    *   Trains the model on the standardized training data.
5.  **Model Evaluation:**
    *   Evaluates the trained model on the standardized testing data to measure its performance (loss and accuracy).
    *   Plots the training and validation accuracy and loss over epochs to visualize the training process.
6.  **Prediction System:**
    *   Demonstrates how to use the trained model to make predictions on new, unseen data.
    *   Takes an example input data point, preprocesses it (standardizes), and uses the model to predict the likelihood of it being malignant or benign.

## Requirements

The project uses common Python libraries for data science and machine learning. You'll need to have the following installed:

*   Python
*   NumPy
*   Pandas
*   Matplotlib
*   Scikit-learn
*   TensorFlow/Keras

These libraries are pre-installed in Google Colab, so if you are running the code there, you don't need to install them separately. If you're running locally, you can install them using pip:
bash pip install numpy pandas matplotlib scikit-learn tensorflow
## How to Run the Code

1.  **Open the Colab Notebook:** Upload the `.ipynb` file to Google Colab or open it directly if it's already there.
2.  **Run the Cells:** Execute each code cell in the notebook sequentially. You can do this by clicking the "Run" button next to each cell or by pressing `Shift + Enter`.

## Understanding the Code

*   **Data Loading:** The code starts by loading the dataset and inspecting its structure and contents using pandas functions like `.head()`, `.tail()`, `.shape()`, `.info()`, `.isnull().sum()`, and `.describe()`.
*   **Preprocessing:** The data is split into features (X) and labels (Y), and then into training and testing sets. The `StandardScaler` is used to normalize the feature data, which is crucial for optimal performance of the neural network.
*   **Neural Network:** A simple `Sequential` model is created with layers. The first layer is `Flatten` to prepare the input. The `Dense` layers are the core of the network, with 'relu' activation for the hidden layer and 'sigmoid' activation for the output layer.
*   **Training:** The `model.compile()` step configures the training process (optimizer, loss function, and metrics). `model.fit()` trains the model on the training data, using a portion for validation.
*   **Evaluation:** `model.evaluate()` measures the model's performance on the unseen test data. The plots help visualize how well the model learned during training.
*   **Prediction:** The code demonstrates how to prepare a new data point and use `model.predict()` to get the model's output (probabilities for each class). `np.argmax()` is used to determine the predicted class (0 for malignant, 1 for benign).

## Results

After training and evaluation, the notebook will output the accuracy of the model on the test data and show an example of how the prediction system works. The plots will show the training and validation progress.

=>Practice Projet
