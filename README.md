# Money-Management-classification

This project focuses on building a machine learning model to automatically classify financial transactions into predefined categories. Additionally, this repository explores unsupervised learning techniques to discover natural spending categories and enhance the dataset.

-----

## Project Overview

This project is a smart money manager app that helps users in Kenya track their spending and achieve their savings goals. A key feature is the ability to categorize transactions to understand spending habits. Currently, this process is partly manual. This project aims to automate the classification of transactions into 13 distinct categories using machine learning, improving user experience and providing valuable financial insights.

###  Project Objectives

  * **Primary Objective:** Develop a high-performing classification model to categorize transactions with high accuracy.
  * **Secondary Objective:** Utilize unsupervised clustering techniques to discover natural groupings in transaction data, which can be used to validate or refine the existing categories.
  * **Tertiary Objective:** Implement anomaly detection to identify fraudulent or unusual transactions.

-----

##  The Business Problem

For a financial management app to be effective, it must provide users with a clear, accurate and effortless overview of their spending habits. Manually categorizing every transaction is tedious and a major point of friction for users. An automated system that can instantly and accurately classify expenditures (e.g., "Safaricom" into "Data & WiFi", "Naivas" into "Groceries") is essential for user retention and for providing deeper, data-driven financial advice.

-----

##  The Data 

The dataset was provided by Alvin for the Zindi competition. It includes transaction records from their beta release, primarily sourced from MPESA SMS receipts.

  * `Train.csv`: Contains transactions with verified, pre-assigned categories. Used to train the machine models.
  * `Test.csv`: Contains transactions without categories. The model's performance is evaluated on this set.
  * `extra_data.csv`: A larger set of unverified, user-classified transactions.

-----

##  Methodology

This project is approached in two main stages: Unsupervised Discovery followed by Supervised Classification.

### Step 1: Unsupervised Learning - Category Discovery

Before training a classifier, we use clustering to understand the inherent structure of the transaction data. This helps validate the predefined categories and can be used to label new data more efficiently.

1.  **Text Preprocessing & Vectorization:** Transaction descriptions (`MERCHANT_NAME`) are cleaned (lowercase, remove punctuation). They are then converted into meaningful numerical vectors using **Sentence-BERT embeddings**, which captures semantic similarity.
2.  **Clustering:** The **DBSCAN** algorithm is applied to the vectors. DBSCAN is chosen for its ability to identify clusters of varying shapes and sizes and its robustness to outliers, which is ideal for noisy transaction data.
3.  **Cluster Analysis:** The resulting clusters are analyzed to identify common themes (e.g., a cluster containing "Uber," "Bolt," "Little Cab"). This helps in assigning a single category label to an entire group of similar transactions, a technique known as semi-supervised learning.

### Stop 2: Supervised Learning - Transaction Classification

1.  **Feature Engineering:**
      * **Text-Based Features:** Features are extracted from the merchant name.
      * **Time-Based Features:** The transaction timestamp is used to create features like `hour_of_day` and `day_of_week`.
2.  **Model Selection:** Several models will be evaluated.
3.  **Model Training & Evaluation:** The model is trained on the `Train.csv` dataset. 

-----

##  Project Structure

```
├── data/
│   ├── raw/
│   │   ├── Train.csv
│   │   ├── Test.csv
│   │   └── extra_data.csv
│   └── processed/
│       └── cleaned_data.csv
│
├── notebooks/
│   ├── 01_Data_Exploration_and_Cleaning.ipynb
│   ├── 02_Unsupervised_Clustering_for_Category_Discovery.ipynb
│   ├── 03_Supervised_Classification_Modeling.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── feature_engineering.py
│
├── models/
│   └── best_model.pkl
│
├── README.md
├── requirements.txt
└── LICENSE
```

-----

## 6\. Getting Started

### Prerequisites

  * Python 3.8+
  * Jupyter Notebook or JupyterLab
  * Access to a terminal or command prompt

### Installation & Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Itskuloba/money-management-classification.git
    cd money-management-classification
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Launch Jupyter and run the notebooks:**
    Navigate through the `notebooks/` directory to see the step-by-step process from data exploration to modeling.

    ```bash
    jupyter notebook
    ```

-----

##  Acknowledgements

  * Thanks to **Zindi** for providing the data and hosting this challenging and insightful competition.
 
