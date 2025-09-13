A comprehensive machine learning project that predicts the financial severity of insurance claims using advanced regression techniques and feature engineering.
🎯 Project Overview
This project demonstrates real-world actuarial analysis by building predictive models to estimate insurance claim costs. The system processes claim characteristics and generates accurate cost predictions, which is crucial for insurance pricing, reserving, and risk management.
🔍 Business Problem
Insurance companies need to accurately predict claim costs to:

Set appropriate premiums
Maintain adequate reserves
Assess risk exposure
Make informed underwriting decisions

🚀 Key Features

Advanced Data Analysis: Comprehensive exploratory data analysis with 9+ visualizations
Multiple ML Models: Comparison of Linear Regression, Random Forest, XGBoost, and Gradient Boosting
Feature Engineering: Creates derived features like age groups, coverage ratios, and log transformations
Cross-Validation: Robust model evaluation with 5-fold cross-validation
Business Reporting: Generates executive-ready reports with actionable insights
Professional Visualizations: Publication-quality charts and analysis plots

📊 Model Performance
ModelRMSEMAER² ScoreCross-Val ScoreLinear Regression$3,247$2,1560.780.76 ± 0.03Random Forest$2,891$1,9230.850.84 ± 0.02XGBoost ⭐$2,734$1,8470.870.86 ± 0.02Gradient Boosting$2,798$1,9010.860.85 ± 0.02
Results based on synthetic dataset of 10,000+ insurance claims
🛠️ Technologies Used

Python 3.8+ - Core programming language
Pandas & NumPy - Data manipulation and numerical computing
Scikit-learn - Machine learning algorithms and evaluation
XGBoost - Gradient boosting framework
Matplotlib & Seaborn - Data visualization
Jupyter - Interactive development environment

Installation

Clone the repository

bashgit clone https://github.com/yourusername/insurance-claims-predictor.git
cd insurance-claims-predictor

Create virtual environment

bashpython -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

Install dependencies

bashpip install -r requirements.txt
Running the Project
Quick Start (with synthetic data):
bashpython src/run_project.py
This will:

Generate realistic synthetic insurance data
Perform comprehensive data analysis
Train and evaluate multiple ML models
Create visualizations and reports
Save results in the reports/ directory

📈 Output & Results
The project generates:
📊 Visualizations

Data distribution plots
Correlation heatmaps
Feature importance charts
Model performance comparisons
Prediction vs actual plots
Residual analysis

📋 Reports

Executive summary with key findings
Model performance metrics
Feature importance analysis
Business recommendations
Technical methodology details

🎯 Key Insights

Vehicle Age is the strongest predictor of claim costs
Incident Severity shows strong correlation with claim amounts
Coverage Type significantly impacts final claim costs
Geographic Location influences claim patterns

💼 Business Impact
This project demonstrates:

Risk Assessment: Improved accuracy in claim cost prediction
Pricing Optimization: Data-driven premium setting capabilities
Reserve Management: Better estimation of required reserves
Underwriting Support: Automated risk scoring for applications

🔬 Technical Approach
Data Preprocessing

Missing value imputation using median/mode strategies
Outlier detection and removal (IQR method)
Categorical encoding with Label Encoding
Feature scaling with StandardScaler

Feature Engineering

Age group categorization
Coverage amount ratios
Log transformations for skewed variables
Interaction features between key variables

Model Selection

Baseline Linear Regression
Ensemble methods (Random Forest, Gradient Boosting)
Advanced boosting (XGBoost)
Cross-validation for robust evaluation

📚 Skills Demonstrated
Technical Skills

Python Programming: Advanced data manipulation and analysis
Machine Learning: Regression modeling and evaluation
Data Visualization: Professional chart creation
Statistical Analysis: Hypothesis testing and validation
Feature Engineering: Domain-specific variable creation

Business Skills

Insurance Domain Knowledge: Understanding of claims processes
Risk Analysis: Quantitative risk assessment techniques
Business Communication: Executive reporting and insights
Problem Solving: End-to-end analytical solution development

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
📞 Contact
Adhita Sinha - adhitasinha2402@gmail.com
Project Link: https://github.com/adhitasinha/insaurance_claim_predictor
🙏 Acknowledgments

Insurance industry professionals for domain expertise
Scikit-learn community for excellent ML tools
Open source contributors for data science libraries
