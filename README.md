#  Email/SMS Spam Classifier

A Machine Learning-based web application that classifies Email and SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and the **Multinomial Naive Bayes** algorithm.

##  Features

- Detects whether a message is **Spam** or **Not Spam**
- Text preprocessing using NLTK
- TF-IDF Vectorization
- Multinomial Naive Bayes Classifier
- Interactive web interface built with Streamlit

##  Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Matplotlib
- Seaborn
- WordCloud
- XGBoost (used for model comparison)

##  Machine Learning Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Text Preprocessing
   - Lowercase conversion
   - Tokenization
   - Stopword removal
   - Stemming
4. TF-IDF Feature Extraction
5. Model Training and Evaluation
6. Model Deployment using Streamlit

##  Models Evaluated

- Gaussian Naive Bayes
- Multinomial Naive Bayes ✅ (Selected Model)
- Bernoulli Naive Bayes
- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- AdaBoost
- Bagging Classifier
- Extra Trees Classifier
- Gradient Boosting
- XGBoost

##  Best Model

**Multinomial Naive Bayes (MNB)** was selected as the final model because it achieved the highest precision while maintaining excellent testing accuracy, making it well-suited for spam classification.


##  Run Locally

Clone the repository:

```bash
git clone https://github.com/your-username/Spam-Classifier.git
```

Navigate to the project folder:

```bash
cd Spam-Classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

##  Deployment

The application is deployed using **Streamlit Community Cloud**.

