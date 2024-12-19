# Credit_Risk_Assessment_Model
An attempt at the bluechip ft DSN Hackathon Credit Assessment Project to predict the risk of loan approvals based on certain features.

## Project Overview
This project aims to build a machine learning model that assesses the credit risk of loan applicants. By leveraging a variety of data preprocessing techniques, feature engineering, and resampling methods, the model predicts whether a loan application is likely to be approved or not.

## Features of the Project
- **Data Preprocessing**: Handles missing values and encodes categorical variables.
- **Feature Engineering**: Creates new features such as income ratios and polynomial features to enhance model performance.
- **Resampling Techniques**: Implements various techniques like SMOTE, ADASYN, and SMOTETomek to address class imbalance in the dataset.
- **Model Training**: Trains an XGBoost classifier and evaluates it using precision, recall, F1-score, and accuracy metrics.
- **Model Persistence**: Saves and loads models using joblib for reproducibility.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abrokinla/Credit_Risk_Assessment_Model.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Credit_Risk_Assessment_Model
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare the data:
   - Confirm that the training dataset in the `credit-worthiness-prediction/` directory.
   - Update the file paths in the script as necessary.
2. Run the `Credit_Risk_Assessment_araoye.ipynb` to train and evaluate the model:
   
3. Save the trained model:
   - Trained models are saved in the models directory upon training completion
4. Use the saved model to make predictions on new data:
   - You can load and use the saved model to predict on the test data provided

## Project Structure
- `credit-worthiness-prediction/`: Contains the dataset files.
- `models/`: Stores the trained models.
- `scripts/`: Includes scripts for preprocessing, training, and prediction.
- `requirements.txt`: Lists all the dependencies required for the project.

## Key Dependencies
- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn

## Contributions
Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request or open an issue.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Bluechip FT DSN Hackathon organizers for providing the dataset and challenge.
- Open-source libraries and contributors for tools used in this project.

