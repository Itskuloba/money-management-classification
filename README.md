# Money-Management-classification

This project focuses on building a machine learning model to automatically classify financial transactions into predefined categories. Additionally, it explores unsupervised learning techniques to discover natural spending categories.
streamlit app link;https://transactionclassifier.streamlit.app/
Medium article link;https://medium.com/@nkuloba58/automating-m-pesa-transaction-categorization-building-the-magic-classifier-for-smarter-personal-c1abc24d4577
-----

## Project Overview  
This project powers a **smart money manager app** tailored for Kenyan users, enabling seamless tracking of M-Pesa transactions and progress toward savings goals. A core feature is **automated, real-time spending categorization.**   

Currently, partial manual categorization creates friction and limits user engagement. This project **fully automates transaction classification** using machine learning, combining **merchant name semantics, transaction context, and user behavior** to deliver instant, accurate and personalized financial clarity — all without user input.

---

## Project Objectives  

**Primary Objective:**  
Develop a **production-ready, high-accuracy multi-class classification model** that automatically assigns one of 8 consolidated, meaningful spending categories to every M-Pesa transaction using merchant name, amount, time, and user metadata.

**Secondary Objective:**  
Apply **unsupervised clustering (KMeans on TF-IDF vectors)** to discover natural groupings in merchant names, validate existing category definitions, and enable **semi-supervised labeling** of new or ambiguous merchants for continuous model improvement.

**Tertiary Objective:**  
Implement **anomaly detection** (using isolation forests or reconstruction error from autoencoders) to flag potentially **fraudulent, erroneous, or financially significant outliers** (e.g., unusually large P2P transfers or rare high-value bills), enhancing user trust and financial safety.

-----

##  The Business Problem

A financial management app succeeds only when users can instantly understand their spending patterns without manual effort. Requiring users to manually categorize every transaction — such as tagging "Safaricom Tunukiwa" as *Data & WiFi* or "Naivas Supermarket" as *Groceries* — creates significant friction, leading to user drop-off and reduced engagement.

An **automated, accurate, and real-time transaction classification system** is critical to:
- Eliminate manual tagging,
- Deliver instant spending insights,
- Enable personalized budgeting and financial advice,
- Drive user retention and long-term adoption.

This project builds a robust machine learning pipeline to automatically classify M-Pesa transactions into meaningful spending categories using merchant names, transaction metadata, and user context — transforming raw payment data into actionable financial intelligence.

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

1. **Text Preprocessing & Vectorization:** Transaction descriptions (MERCHANT_NAME) are cleaned (lowercase, remove non-alphabetic characters, normalize spaces). They are then converted into a TF-IDF matrix using TfidfVectorizer (with English stop words removed), resulting in a sparse matrix that is converted to a dense array for clustering.
2. **Clustering:** The KMeans algorithm is applied to the dense TF-IDF vectors. To determine the optimal number of clusters, the elbow method is used by fitting KMeans models for k values ranging from 10 to 25 (with 10 initializations per run for stability), calculating inertia (within-cluster sum of squares), and plotting the results to identify the "elbow" point where inertia decreases more slowly.
3. **Cluster Analysis:** The resulting clusters are analyzed to identify common themes (e.g., a cluster containing "Uber," "Bolt," "Little Cab"). This helps in assigning a single category label to an entire group of similar transactions, a technique known as semi-supervised learning. Additional clustering options like Gaussian Mixture Models or DBSCAN are available for comparison, with silhouette scores potentially used for evaluation.


### Step 2: Supervised Learning - Transaction Classification

Label Preprocessing: Category labels are cleaned by assigning specific merchants to consistent categories (e.g., supermarkets like 'NAIVAS', 'CARREFOUR' to 'Groceries'; utilities like 'KPLC', 'ZUKU' to 'Bills & Fees'; restaurants like 'JAVA', 'CAFE' to 'Going out'). Peer-to-peer transactions are detected using heuristics on merchant names (e.g., checking for personal name patterns via regex, absence of business indicators, digits, or full uppercase) and assigned to 'Family & Friends'. Rare categories (e.g., 'Education', 'Health', 'Emergency fund', 'Rent / Mortgage', 'Loan Repayment') are merged into broader ones like 'Bills & Fees' or 'Miscellaneous' to reduce sparsity and improve model generalization.

**1. Feature Engineering:**

**a)Text-Based Features:** Merchant names are cleaned (lowercased, remove non-alphanumeric) and normalized (uppercased variant). Binary flags are created for keyword matches indicating categories (e.g., data_keywords like 'tunukiwa', 'airtime'; bill_keywords like 'kplc', 'zuku'; supermarket_keywords like 'naivas', 'carrefour'; etc. for banks, transport, loans, health). Strong overrides ensure correct flags for common cases. Merchant frequency is mapped from value counts. Log transformations are applied to 'PURCHASE_VALUE' and 'USER_INCOME'.
**b)Time-Based Features:** The transaction timestamp ('PURCHASED_AT') is parsed to create features like 'purchase_hour', 'purchase_day_of_week', 'purchase_month', 'is_weekend' (binary), and 'part_of_day' (categorized as 'morning', 'afternoon', 'evening', 'night' with one-hot encoding).
**c)Other Features:** Binary flag for P2P payments. 'USER_GENDER' is one-hot encoded. 'USER_AGE' is imputed. 'USER_HOUSEHOLD' and other numerics are included. Missing values are imputed with median using SimpleImputer. Specific columns (e.g., age, household, hour, logs) are scaled with StandardScaler.

**2. Model Selection:** Several models are evaluated, including Logistic Regression, Random Forest, XGBoost, and CatBoost.

**3. Model Training & Evaluation:** The dataset is split 80/20 stratified. Class weights are computed for imbalance and applied (sample weights for XGBoost). Hyperparameters are tuned via RandomizedSearchCV and GridSearchCV on macro F1. Models are compared on train/val accuracy, macro F1, precision, recall. Classification reports and confusion matrices are generated. Learning curves (F1 and accuracy) are plotted using cross-validation to assess bias/variance, revealing high overfitting. Misclassified transactions are analyzed in a report showing merchant name, value, true/predicted labels. The best model by val F1 is saved, along with preprocessor artifacts (imputer, scaler, label encoder, etc.).

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
 
