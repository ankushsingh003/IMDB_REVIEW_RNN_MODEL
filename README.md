# Movie Review Sentiment Analysis with SimpleRNN

This project implements a sentiment analysis model for movie reviews using a Simple Recurrent Neural Network (RNN) trained on the IMDB dataset. The model predicts whether a movie review is positive or negative.

## Features

- Pre-trained SimpleRNN model for sentiment classification
- Web-based interface using Streamlit for easy prediction
- Jupyter notebooks for training and testing the model

## Files

- `imdb.ipynb`: Jupyter notebook for training the RNN model on IMDB data
- `prediction.ipynb`: Jupyter notebook for testing predictions on sample reviews
- `main.py`: Streamlit application for interactive sentiment analysis
- `imdb_simpleRNN_model.h5`: Pre-trained model file (generated after training)

## Setup Instructions

### Prerequisites

- Python 3.70 or higher
- pip (Python package installer)

### Installation

1. Clone or download this repository to your local machine.

2. Navigate to the project directory:

   ```
   cd path/to/MOVIE_REVIEW_RNN_MODEL
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

   This will install:
   - numpy
   - pandas
   - tensorflow
   - streamlit

## Usage

### Training the Model (Optional)

If you want to retrain the model:

1. Open `SimpleRNN/imdb.ipynb` in Jupyter Notebook or JupyterLab.
2. Run all cells in the notebook. This will:
   - Load the IMDB dataset
   - Preprocess the data
   - Build and train the SimpleRNN model
   - Save the trained model as `imdb_simpleRNN_model.h5`

### Running the Sentiment Analysis App

1. Navigate to the SimpleRNN folder:

   ```
   cd SimpleRNN
   ```

2. Run the Streamlit application:

   ```
   streamlit run main.py
   ```

3. Open your web browser and go to the URL displayed (usually `http://localhost:8501`).

4. Enter a movie review in the text area and click "Predict Sentiment" to see the analysis.

### Testing Predictions (Alternative)

1. Open `SimpleRNN/prediction.ipynb` in Jupyter Notebook.
2. Run the cells to load the model and test functions.
3. Modify the `sample_review` variable to test different reviews.

## Model Details

- **Architecture**: Embedding layer -> SimpleRNN layer -> Dense output layer
- **Input**: Movie review text (preprocessed to sequences of word indices)
- **Output**: Sentiment prediction (Positive/Negative) with confidence score
- **Dataset**: IMDB movie reviews (25,000 training, 25,000 test samples)
- **Vocabulary size**: 10,000 most frequent words
- **Sequence length**: Padded/truncated to 500 words

## Troubleshooting

- **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
- **Model loading issues**: Make sure `imdb_simpleRNN_model.h5` is in the same directory as the scripts
- **Streamlit not starting**: Check that port 8501 is available or specify a different port with `streamlit run main.py --server.port 8502`
- **Memory issues**: If training fails due to memory, reduce batch size in `imdb.ipynb`

## License

This project is for educational purposes. The IMDB dataset is provided by Keras/TensorFlow.
