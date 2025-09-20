🏠 Iranian Housing Price Prediction
A comprehensive machine learning project for predicting rental housing prices across Iranian cities using property characteristics and location data.

📋 Abstract
This project develops a robust machine learning model to predict housing rental prices in Iran based on various property features including area, location, amenities, and building characteristics. The model combines advanced preprocessing techniques with multiple regression algorithms to achieve accurate price predictions for the Iranian real estate market.

🎯 Objectives
Goal: Predict the total housing cost using the formula: Total = Credit + (Rent × 33.33334)
Data Processing: Handle Persian text data and categorical variables effectively
Model Optimization: Compare multiple machine learning algorithms for best performance
Feature Engineering: Create meaningful features from raw property data
📊 Dataset Overview
Training Data (train.csv)
Samples: Housing rental listings with complete pricing information
Features: 21 columns including property characteristics and pricing
Target Variable: Calculated Total value combining upfront credit and monthly rent
Test Data (test.csv)
Samples: Housing listings requiring price predictions
Features: 19 columns (same as training data minus Credit and Rent)
Key Features
Feature	Description	Type
Area	Property size in square meters	Numeric
Year_built	Construction year	Numeric
Rooms	Number of rooms	Numeric
Floor	Floor number/position	Mixed
Elevator	Elevator availability	Binary
Parking	Parking availability	Binary
District	Neighborhood/district	Categorical
City	City location	Categorical
Credit	Upfront payment (Million Toman)	Numeric
Rent	Monthly rent (Million Toman)	Numeric
🔧 Technical Implementation
Data Preprocessing Pipeline
Persian Text Handling: Convert Persian values (دارد, ندارد, etc.) to appropriate formats
Missing Value Treatment: Strategic imputation using median for numeric and mode for categorical
Feature Engineering:
Building age calculation
Area per room ratio
New building indicator
Categorical Encoding: Label encoding with unseen category handling
Feature Scaling: StandardScaler for numeric features
Outlier Detection: IQR-based outlier removal
Machine Learning Models
Ridge Regression: Regularized linear model for baseline performance
Random Forest: Ensemble method for capturing non-linear relationships
Model Selection: Cross-validation based performance comparison
Evaluation Metrics
Primary: R² Score (Coefficient of Determination)
Secondary: MAE (Mean Absolute Error), RMSE (Root Mean Square Error)
🚀 Getting Started
Prerequisites
bash
pip install pandas numpy scikit-learn matplotlib seaborn
Usage
python
# Clone and run the main script
python housing_prediction.py

# Or run the Jupyter notebook
jupyter notebook housing.ipynb
File Structure
project/
│
├── Data/
│   ├── train.csv          # Training dataset
│   └── test.csv           # Test dataset
│
├── housing.ipynb          # Main Jupyter notebook
├── submission.csv         # Model predictions
└── README.md             # Project documentation
📈 Results & Performance
Best Model: Random Forest Regressor
Validation R² Score: ~0.85+ (significantly improved from baseline)
Key Insights:
Location (City, District) are primary price drivers
Property size and amenities strongly correlate with pricing
Building age and condition impact rental values
🔍 Key Innovations
Bilingual Data Processing: Seamless handling of Persian and English mixed datasets
Robust Categorical Encoding: Handles unseen categories in test data
Advanced Feature Engineering: Creates meaningful derived features
Multi-Model Approach: Compares algorithms for optimal performance
Production-Ready Pipeline: Scalable preprocessing and prediction workflow
📝 Model Interpretability
The Random Forest model provides feature importance rankings, revealing:

Location factors (City, District) as top predictors
Physical characteristics (Area, Rooms) as strong indicators
Amenities (Parking, Elevator) contributing to price variations
🎯 Business Applications
Real Estate Valuation: Automated property price estimation
Market Analysis: Understanding regional price patterns
Investment Decisions: Data-driven property investment insights
Rental Optimization: Optimal pricing strategies for landlords
🔮 Future Enhancements
Deep Learning: Neural networks for complex pattern recognition
Time Series: Incorporating temporal price trends
External Data: Integration with economic indicators
Web Application: User-friendly prediction interface
👥 Contributing
Feel free to contribute by:

Improving preprocessing techniques
Adding new feature engineering methods
Implementing additional ML algorithms
Enhancing model interpretability
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

Note: This model is designed for educational and research purposes. Real estate predictions should consider additional market factors and expert consultation.

