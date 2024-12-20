 # GROUP - 8 : Fake News Detection Project ( English & telugu)

## Technologies Used
- Python (requests, BeautifulSoup, pandas, ast, spacy, counter, csv, math, sklearn, numpy,flask,scikit-learn)
- Jupyter Notebook for code development and data processing

## Data Collection
- Webscrapping from news articles

## Preprocessing Steps

Our preprocessing pipeline includes:
1. Tokenization  
2. Part-of-Speech (POS) Tagging  
3. Stop Words Removal  
4. Lemmatization and Stemming  
5. Named Entity Recognition (NER)  


## Feature Enginnering

1. TF-IDF Scoring  
2. N-Gram Generation  
3. Feature Extraction
4. Feature Selection


## Model Building

Our Modl building includes two models :
1. KNN Classifier (K-Nearest Neighbour)
2. SVM Model (Support Vector Machine)
3. Logistic Regression 

## How the Code Works

1. **Pre-Processing**: 
    - **Tokenization**: Split text into individual tokens using SpaCy's tokenizer.  
    - **Part-of-Speech (POS) Tagging**: Tagged tokens with their respective parts of speech using SpaCy.  
    - **Stop Words Removal**: Removed commonly used words with little meaning using SpaCy's built-in stop word list.  
    - **Lemmatization and Stemming**: Reduced words to their base form using SpaCy for lemmatization and applied a custom stemming function.
    - **Named Entity Recognition (NER)**: Identified entities such as names, locations, and organizations using SpaCy's NER module.  
      

2. **Feature Engineering**:  
    - **TF-IDF Scoring**: Calculated Term Frequency-Inverse Document Frequency (TF-IDF) scores for feature importance.  
    - **N-Gram Generation**: Created sequences of n words (n-grams) for more robust text representation.
    - **Feature Extraction**: Extracted important features from the processed text using TF-IDF scores.  
    - **Feature Selection**: Selected the most relevant features using statistical techniques to reduce dimensionality.  

3. **Model Building**:  
    - **KNN Classifier (K-Nearest Neighbour)**: Implemented a KNN classifier to predict the class labels based on similarity to neighboring data points.  
    - **SVM Model (Support Vector Machine)**: Trained an SVM model to classify news articles into fake or real using a hyperplane for separation.  
    - **Logistic Regression**: - Trained a Logistic Regression model to classify news articles by estimating probabilities and applying a decision boundary for classification.  

## Running the Code

### Requirements
- Python 3.x
- Jupyter Notebook
- Required libraries:
  - `requests`
  - `beautifulsoup4`
  - `pandas`
  - `flask`
  - `scikit-learn`
  - `ast`
  - `spacy`
  - `counter`
  - `csv`
  - `math`
  - `sklearn`
  - `numpy`

## Installation  

To run the Fake News Detection model, install the required Python libraries using the following commands:  

```bash
pip install requests
pip install beautifulsoup4
pip install pandas
pip install spacy
pip install numpy
pip install scikit-learn
pip install scikit-learn
pip install flask
```

## Additional libraries that may not require installation (built-in):
- ast
- csv
- math
- collections (Counter)

## To run the UI Model in web:
```bash
    python app.py
```
## Then go to :
http://127.0.0.1:5000/

## and use it