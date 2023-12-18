### **GPU Data Race Detection Project**

#### Background:
This project aims to address data race issues within GPU kernels, a significant challenge in parallel computing. Data races arise from concurrent access to shared resources by multiple threads, causing erratic behavior and debugging challenges.

#### Objective:
The primary goal is to improve data race detection in GPU kernels using a predictive modeling approach with supervised learning. This novel method focuses on identifying patterns in labeled data race instances using techniques like CNNs (Convolutional Neural Networks) and Regression Models.

#### Key Features:
- **Supervised Learning Technique:** Predicts critical code segments prone to race conditions using labeled data.
- **GPU Memory Hierarchies:** Identifies susceptible areas like atomics, barriers, and mutexes to navigate complexities.
- **Evaluation Metrics:** Measures model performance using precision, recall, and accuracy.
- **Model Variants Comparison:** Compares CNN against RNN, LSTM, and GRU for data race detection accuracy.
- **Architecture Modifications:** Iterative adjustments to the CNN architecture for enhanced accuracy.

#### Results:
- **Accuracy Ranges:** Achieved accuracy between 81.34% to 84.21% on the test set.
- **Dataset Evaluation:** Varied accuracies across OpenMP and POSIX datasets due to differing training samples.
- **Detection Efficacy:** Low false positives and negatives in identifying buggy and bug-free files.
- **Model Limitations:** Performance variations due to dataset size and granularity of ASTs (Abstract Syntax Trees).

#### Conclusion:
- **Advancements:** Offers a more efficient means for GPU programmers to identify and mitigate concurrency issues.
- **Potential Scenarios:** Demonstrated effectiveness in detecting specific synchronization patterns and potential for broader applications with appropriate training datasets.
- **Future Scope:** Further improvements anticipated with larger and more diverse training sets for different data race patterns.

#### Additional Notes:
- **Microbenchmark Evaluation:** Successful detection in microbenchmarks representative of specific data race patterns.
- **Comparative Analysis:** Contrasts the model's capabilities against existing debugging tools and race detectors.


## Overview

models.py provides various utility functions and preprocessing steps necessary for AST and source code data analysis.

## How to Run

To use this script, follow these steps:

1. Mount Google Drive to access the data files.
2. Set up the necessary data paths.
3. Utilize the functions provided for data preprocessing and analysis.

### Code Structure

The code is structured into several sections:

- **Data Preprocessing Step: Utility Functions**: Contains functions for data preprocessing.
  - `cleaned_string(input_string)`: Preprocesses an input string by removing leading and trailing whitespace and converting it to lowercase.
  - `load_data_and_labels()`: Loads data from files, tokenizes it, and generates labels.
  - `pad_token_vectors(token_vectors, padding_word="PAD")`: Pads token vectors to the same length.
  - `average_length(sentences, padding_word="PAD")`: Pads token vectors based on the calculated or overridden average length.
  - `build_vocabulary_mapping(sentences)`: Builds a vocabulary mapping from token to index.
  - `build_input_data(sentences, labels, vocabulary)`: Maps token vectors and labels to vectors based on a vocabulary.
  - `save_processed_data(...)`: Saves various data structures as pickle files.

### Functions and Their Functionalities

- `cleaned_string(input_string)`: Preprocesses an input string by removing leading and trailing whitespace and converting it to lowercase.

- `load_data_and_labels()`: Loads data from files, tokenizes it, and generates labels.

- `pad_token_vectors(token_vectors, padding_word="PAD")`: Pads token vectors to the same length. The length is defined by the longest token vector.

- `average_length(sentences, padding_word="PAD")`: Pads token vectors to the same length based on the calculated average length of token vectors.

- `build_vocabulary_mapping(sentences)`: Builds a vocabulary mapping from token to index based on the vector.

- `build_input_data(sentences, labels, vocabulary)`: Maps token vectors and labels to vectors based on a vocabulary.

- `save_processed_data(...)`: Saves various data structures as pickle files in the specified directory.

## How to Use Functions

1. Load and preprocess data using `load_data_and_labels()` function.
2. Pad token vectors using `pad_token_vectors()` or `average_length()` functions.
3. Build vocabulary mapping using `build_vocabulary_mapping()` function.
4. Map token vectors and labels using `build_input_data()` function.
5. Save processed data using `save_processed_data()` function.

## Important Notes

- Ensure data paths are correctly set before executing any functions.
- Refer to function docstrings for detailed information about arguments and return values.

### Load Previous Saved Data

- **`load_prev_saved_data()`**:
  - Loads previously saved data from pickle files.
  - Arguments:
    - `training_data_dest` (str): Directory path containing saved data files.
  - Returns:
    - `x` (list or numpy.ndarray): Mapped token vectors.
    - `y` (list or numpy.ndarray): Labels.
    - `vocabulary` (dict): Vocabulary mapping.
    - `vocabulary_inv` (list): Inverse vocabulary list.

### Load Test Data

- **`load_test_data()`**:
  - Loads test data for evaluation.
  - Arguments:
    - `training_data_dest` (str): Directory path containing saved data files.
  - Returns:
    - `x` (numpy.ndarray): Input data (token vectors).
    - `y` (numpy.ndarray): Target labels.
    - `vocabulary` (dict): Vocabulary mapping.
    - `vocabulary_inv` (list): Inverse vocabulary list.

### Load Data

- **`load_data(avg_len=False, load_saved_data=False, load_testdata=False)`**:
  - Loads and preprocesses data for the dataset.
  - Arguments:
    - `avg_len` (bool): Determines whether to use average length or padding for sentences.
    - `load_saved_data` (bool): Flag for loading previously saved data.
    - `load_testdata` (bool): Flag for loading test data.
  - Returns:
    - `x` (list or numpy.ndarray): Mapped token vectors.
    - `y` (list or numpy.ndarray): Labels.
    - `vocabulary` (dict): Vocabulary mapping.
    - `vocabulary_inv` (list): Inverse vocabulary list.

### Reading Functions

- **`read_ast(file_name)`**:
  - Reads an Abstract Syntax Tree (AST) from a file.
  - Arguments:
    - `file_name` (str): Name of the file containing the AST.
  - Returns:
    - `list`: List of strings representing lines in the AST.

- **`read_source_code_file(file_name)`**:
  - Reads a source code file.
  - Arguments:
    - `file_name` (str): Name of the source code file.
  - Returns:
    - `list`: List of strings representing lines in the source code file.

### Source Line Impact Functions

- **`extract_source_line_impacts(ast_list, line_impacts, method="only_parent")`**:
  - Creates a dictionary containing probabilities/impacts for source code lines from an AST.
  - Arguments:
    - `ast_list` (list): List representing the AST.
    - `line_impacts` (dict): Dictionary with AST line numbers and their respective impact probabilities.
    - `method` (str): Method to determine line values: 'only_parent', 'consider_children', 'maximum', or 'average' (default: 'only_parent').
  - Returns:
    - `dict`: Dictionary containing probabilities/impacts for source code lines.

### Highlight Source Code

- **`highlight_source_code(filename, source_code_intrst_lines, method)`**:
  - Highlights specific lines in a source code file based on provided probabilities.
  - Arguments:
    - `filename` (str): Name of the source code file.
    - `source_code_intrst_lines` (dict): Dictionary with line numbers and their associated probabilities.
    - `method` (str): Method used to highlight lines (e.g., 'maximum', 'average').
  - Writes:
    - An HTML file containing the highlighted source code.

**Important Notes**:
- Ensure correct paths and arguments are provided to these functions.
- Refer to the docstrings within the code for detailed information about arguments and return values.

# Deep Learning Models Implementation

This Python script implements various Deep Learning models like CNN, RNN, LSTM, and GRU using Keras with TensorFlow backend for a classification task.

## File Structure

- `data_processing.py`: Contains functions for data preprocessing and loading.
- `model_training.py`: Defines and trains CNN, RNN, LSTM, and GRU models.
- `visualization.py`: Visualizes model performance using learning curves.

## Important Functions

### `load_prev_saved_data()` & `load_test_data()`
- Loads previously saved data or test data for evaluation.
- Parameters: `training_data_dest` (directory path)
- Returns: Input data, labels, vocabulary, inverse vocabulary.

### `load_data()`
- Loads and preprocesses data for the dataset.
- Parameters: `avg_len` (boolean), `load_saved_data` (boolean), `load_testdata` (boolean)
- Returns: Input vectors, labels, vocabulary, inverse vocabulary.

### `cnn_model()`, `rnn_model()`, `lstm_model()`, `gru_model()`
- Define CNN, RNN, LSTM, GRU architectures respectively.
- Parameters: Input data, labels, vocabulary size, sequence length, embedding dimensions, etc.
- Returns: Compiled model for training.

### `train_model()`
- Trains the specified model on given training data.
- Parameters: Model, input data, labels, epochs, batch size, model name.
- Returns: Trained model and training history.

### `plot()`
- Visualizes learning curves (loss and metrics) of the trained model.

## Running the Script

1. **Data Preparation**: Ensure data is available in appropriate directories.
2. **Specify Parameters**: Adjust parameters like epochs, batch size, etc., if needed.
3. **Run the Script**:
   - Execute the script in a Python environment.
   - Example: `python model_training.py`

## Notes
- Make sure to install the required dependencies like TensorFlow, Keras, NumPy, Matplotlib, etc., beforehand.
- Adjust data paths, hyperparameters, and configurations based on specific requirements.
