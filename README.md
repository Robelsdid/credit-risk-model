# Credit Risk Probability Model for Alternative Data

## Overview
This project implements an end-to-end credit risk scoring model for Bati Bank, leveraging alternative data from an eCommerce platform to enable buy-now-pay-later services. The solution covers data processing, feature engineering, model training, deployment, and automation, following best practices in MLOps and regulatory compliance.

# i tried to summurise it

## Credit Scoring Business Understanding 

1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel Accords (I, II, and III) are a set of international banking regulations developed by the Basel Committee on Banking Supervision to ensure the stability and soundness of the global financial system. Basel II, in particular, introduced a more risk-sensitive framework for measuring credit risk, requiring banks to adopt advanced methods for risk assessment and capital adequacy. This regulatory environment means that banks must not only measure risk accurately but also demonstrate to regulators that their models are transparent, interpretable, and well-documented.

Interpretability is crucial because regulators, auditors, and internal stakeholders must understand how risk scores are derived, especially when using internal ratings-based (IRB) approaches.

Documentation is required to ensure that model assumptions, data sources, and methodologies are clear and auditable.
The three pillars of Basel II—minimum capital requirements, supervisory review, and market discipline—demand that banks disclose relevant information and justify their risk management practices.

Basel III further strengthens these requirements by increasing capital and liquidity standards, introducing new buffers, and emphasizing the need for robust risk governance and transparency.

The Basel Accords drive the need for interpretable and well-documented models to ensure regulatory compliance, facilitate supervisory review, and promote market discipline. This is especially important as models become more complex and as banks are held accountable for their risk management practices.

2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In many real-world datasets, especially those involving alternative data sources, a direct "default" label (i.e., a clear indicator of whether a customer failed to repay a loan) may not be available. In such cases, it is necessary to define a proxy variable—an observable outcome that approximates default behavior (e.g., late payments, account inactivity, or other adverse events).

Necessity: Without a proxy, it is impossible to train or validate a supervised learning model for credit risk. The proxy serves as a stand-in for the true outcome of interest.

Risks: If the proxy does not accurately reflect true default risk, the model may misclassify customers. This can lead to:
False positives: Good customers being denied credit (missed business opportunities).
False negatives: High-risk customers being approved (increased losses).

The Basel Accords highlight the importance of accurate risk measurement and model validation. Using a poorly chosen proxy can undermine the effectiveness of risk management and expose the bank to regulatory and financial risks.

Creating a proxy variable is essential when a direct default label is unavailable, but it introduces risks related to model accuracy and business outcomes. Careful selection, validation, and documentation of the proxy are critical to minimize these risks.

3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

The Basel Accords require banks to balance predictive performance with interpretability and regulatory compliance:
Simple, interpretable models (e.g., Logistic Regression with Weight of Evidence encoding):
Advantages: Easy to explain, audit, and validate; align well with regulatory expectations for transparency and accountability.

Disadvantages: May not capture complex, nonlinear relationships in the data, potentially limiting predictive accuracy.

Complex, high-performance models (e.g., Gradient Boosting, Neural Networks):
Advantages: Can model intricate patterns and interactions, often resulting in higher predictive power and better risk differentiation.

Disadvantages: Often seen as "black boxes," making them harder to interpret, justify, and monitor. This can be problematic in regulatory reviews and may increase model risk.

-The Basel framework emphasizes the need for robust model governance, validation, and ongoing monitoring, regardless of model complexity. Banks must be able to explain and defend their models to supervisors and the public.

The key trade-off is between interpretability (and regulatory ease) and predictive performance. In highly regulated environments, the ability to explain and justify model decisions is often as important as, or more important than, marginal gains in predictive accuracy.

-Basel I introduced the concept of risk-weighted assets and minimum capital requirements, laying the foundation for international banking regulation.

-Basel II enhanced risk sensitivity, introduced internal ratings-based approaches, and emphasized supervisory review and market discipline.

-Basel III responded to the global financial crisis by raising capital and liquidity standards, introducing new buffers, and addressing systemic risk and procyclicality.

-The Basel Accords have evolved to address the increasing complexity of banking, the need for better risk measurement, and the importance of transparency and disclosure.

-Challenges include implementation complexity, regulatory inconsistency, procyclicality, and unintended consequences such as credit rationing and increased compliance costs.

## Project Structure
```
credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
``` 