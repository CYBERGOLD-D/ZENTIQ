ZENTIQ – Nigerian Retail Support

Machine Learning-Powered Customer Support Assistant for Nigerian Retail Businesses

1. Project Overview

ZENTIQ – Nigerian Retail Support is a machine learning project designed to provide quick and relevant answers to common questions raised by Nigerian retail business owners.

The system uses Natural Language Processing (NLP) and machine learning to identify the intent behind a customer's question and return an appropriate predefined response.

It aims to support small and medium-sized retail businesses by making information about sales, payments, inventory, deliveries, and other business operations easier to access.

2. Project Objectives

- Develop an intelligent retail support assistant.
- Classify customer questions into relevant support categories.
- Provide appropriate responses to common retail business questions.
- Apply NLP techniques to process customer queries.
- Evaluate the performance of a machine learning classification model.
- Build an interactive chatbot interface using Streamlit.

3. Technologies Used

- Python – Main programming language.
- Pandas – Data processing and manipulation.
- NumPy – Numerical operations.
- Scikit-learn – Machine learning and model evaluation.
- TF-IDF Vectorizer – Text feature extraction.
- Logistic Regression – Intent classification.
- Joblib – Model serialization and loading.
- Streamlit – Interactive chatbot interface.
- Matplotlib and Seaborn – Data visualization.
- Jupyter Notebook / VS Code – Development environment.

4. Project Structure

ZENTIQ/
│
├── ai_engine/
├── analytics/
├── assets/
├── automation/
├── database/
├── datasets/
│   ├── raw/
│   └── processed/
├── documents/
├── logs/
├── models/
│   ├── zentIQ_tfidf_vectorizer.pkl
│   ├── zentIQ_logistic_regression_model.pkl
│   └── zentIQ_intent_response_mapping.json
├── modules/
├── notebooks/
├── safety/
├── tests/
├── training/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example

5. Dataset Description

The project contains two prepared datasets:

Nigerian Retail Support Dataset

- 5,000 records.
- 19 customer support intents.
- 11 retail support categories.
- Questions and predefined responses covering common retail business needs.

Payment Questions Dataset

- 264 payment-related questions.
- Prepared for further improvement and integration.

The initial model training and evaluation were conducted using the 5,000-record retail support dataset.

6. Machine Learning Methodology

The project follows these steps:

1. Data collection and preparation.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis (EDA).
4. Intent classification and label preparation.
5. Stratified splitting into training, validation, and testing sets.
6. Text feature extraction using TF-IDF.
7. Model training using Logistic Regression.
8. Model evaluation using classification metrics.
9. Model serialization and integration into the chatbot application.

Dataset Split

Dataset| Records
Training| 3,500
Validation| 750
Testing| 750
Total| 5,000

Model Performance

The Logistic Regression model achieved the following results:

Evaluation Metric| Result
Validation Accuracy| 99.87%
Test Accuracy| 99.60%

The results indicate strong classification performance on the prepared validation and test datasets. Further testing with new, real-world customer questions is recommended to assess generalization.

7. How the Chatbot Works

1. A user enters a retail-related question.
2. The TF-IDF vectorizer converts the question into numerical features.
3. The trained Logistic Regression model predicts the question's intent.
4. The application retrieves the corresponding predefined response.
5. The chatbot displays the response and prediction confidence.

Note: ZENTIQ currently uses an intent-classification model with predefined responses. It is not a generative AI chatbot.

8. Installation and Setup

Step 1: Clone the Repository

git clone https://github.com/CYBERGOLD-D/ZENTIQ/commits?author=CYBERGOLD-D

Step 2: Navigate to the Project Folder

cd zentIQ-nigerian-retail-support

Step 3: Create a Virtual Environment

python -m venv venv

Step 4: Activate the Virtual Environment

Windows PowerShell:

.\venv\Scripts\Activate.ps1

Step 5: Install Dependencies

python -m pip install -r requirements.txt

Step 6: Run the Application

streamlit run app.py

The application will open in your browser.

9. Supported Retail Areas

The chatbot is designed to assist with questions relating to:

- Business Expenses
- Customer Acquisition
- Daily Sales
- Damaged Products
- Delivery Costs and Delays
- Inventory Reordering
- Order Tracking
- Payment Confirmation
- Payment Methods
- POS Issues
- Profit Calculation
- Record Keeping
- Refund Requests
- Selling Prices
- Social Media Marketing
- Stock Availability
- Supplier Onboarding
- Supplier Price Changes

10. Limitations and Future Improvements

Current limitations include:

- Responses are retrieved from predefined templates.
- The payment questions dataset requires further response refinement before model integration.
- The model may confuse closely related payment intents.
- Performance should be tested with additional real-world customer questions.

Future improvements may include:

- Expanding the training dataset.
- Improving payment-related intent classification.
- Adding more diverse response templates.
- Integrating a database for conversation records.
- Introducing confidence thresholds and fallback responses.
- Adding multilingual support for Nigerian users.
- Exploring generative AI integration.
- Deploying the application online for wider access.

11. Project Significance

ZENTIQ demonstrates how machine learning and Natural Language Processing can be applied to practical business support challenges.

By providing an accessible question-and-answer interface, the project explores how technology can help Nigerian retail businesses access useful operational information and improve customer support experiences.

12. Author

Project Name: ZENTIQ – Nigerian Retail Support
Project Type: Machine Learning / Natural Language Processing
Application: Retail Customer Support Chatbot
Development Language: Python

---

ZENTIQ – Making Retail Support Smarter, Simpler and More Accessible.

## Live Application

Try ZENTIQ Nigerian Retail Support here:

[Launch ZENTIQ Chatbot](https://6wm3xmhxvsyxeumf3rkapp9.streamlit.app/)