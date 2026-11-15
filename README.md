# House Price Prediction System

A beginner-friendly Python project that explores housing data, prepares features, trains a Linear Regression pipeline, evaluates it, and provides an interactive command-line prediction application.

## Objectives

- Inspect house characteristics and their relationship to the target price.
- Prepare mixed numeric and categorical features without leaking test data into training.
- Train and evaluate a Linear Regression baseline.
- Accept house details in `main.py` and return a price estimate from the saved model.

## Dataset

The project uses `dataset/Housing.csv`, a public GitHub-hosted copy of the Housing price dataset:

- [Repository](https://github.com/Swarnit04/Housing)
- [CSV file](https://raw.githubusercontent.com/Swarnit04/Housing/44501c9d5735cb2d4d45bf496632df807746e0d2/Housing.csv)

The file contains 545 rows and 13 columns. The target is `price`. The 12 input features are:

`area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, and `furnishingstatus`.

There is no named location column. `prefarea` is a yes/no indicator for a preferred area, not a city or address. Prices are shown in the dataset's units; this project does not assume a currency. Check the source repository's terms before redistributing the dataset.

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- joblib
- Jupyter Notebook

## Project structure

```text
House_Price_Prediction/
├── dataset/
│   └── Housing.csv
├── model/
│   └── house_price_model.pkl
├── notebooks/
│   ├── house_price_analysis.ipynb
│   ├── data_preprocessing.ipynb
│   ├── data_visualization.ipynb
│   ├── linear_regression_model.ipynb
│   └── model_evaluation.ipynb
├── outputs/
│   └── visualizations/
├── src/                 # Reserved for future Python modules
├── visualizations/      # Empty project placeholder; charts are in outputs/visualizations/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

The virtual environment is local to each computer and is excluded from Git. Generated charts are kept in `outputs/visualizations/`.

## Exploratory Data Analysis

`notebooks/house_price_analysis.ipynb` displays the first and last rows, column names and data types, dataset information, descriptive statistics, missing values, duplicate rows, and unique categorical values. The dataset has no missing values or exact duplicates. `price` is the target column.

## Data cleaning and preprocessing

`notebooks/data_preprocessing.ipynb`:

- Removes exact duplicate rows and rows missing the target only if present.
- Checks basic numeric validity while keeping zero parking spaces valid.
- Standardizes categorical text and flags unexpected category values for review.
- Splits the data into 80% training and 20% testing with `random_state=42`.
- Uses a scikit-learn `Pipeline` and `ColumnTransformer`. Numeric feature gaps use the median; categorical gaps use the most frequent value; categorical features are one-hot encoded.
- Fits imputation and encoding steps on training data only.

No rows needed removal in the supplied dataset.

## Visualizations

`notebooks/data_visualization.ipynb` creates and saves:

- House price distribution histogram
- Area vs. price scatter plot
- Numeric-feature correlation heatmap
- House price box plot
- Average price by `prefarea`
- Pair plot for selected numeric features

The model development and evaluation notebooks also save actual-vs-predicted and residual plots in `outputs/visualizations/`.

## Linear Regression model

`notebooks/linear_regression_model.ipynb` trains scikit-learn's `LinearRegression` inside the preprocessing pipeline. The saved file `model/house_price_model.pkl` contains the fitted preprocessing steps and regression model together. The test data is held out from fitting.

## Model evaluation

`notebooks/model_evaluation.ipynb` loads the saved pipeline, recreates the same test split, and calculates metrics on test predictions:

| Metric | Result |
| --- | ---: |
| Mean Absolute Error (MAE) | 970,043.40 |
| Mean Squared Error (MSE) | 1,754,318,687,330.66 |
| Root Mean Squared Error (RMSE) | 1,324,506.96 |
| R-squared (R²) | 0.6529 |

The R² score indicates that the model explains about 65.3% of the variation in held-out prices. MAE is about 19.4% of the mean test price, so predictions have meaningful average error and should be treated as a baseline estimate rather than a precise valuation. RMSE being higher than MAE reflects the stronger effect of larger errors.

## Install dependencies

Open a terminal in the project root. For a fresh checkout, create and activate a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, allow it for the current terminal only and activate again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency list includes Jupyter and the libraries used by the notebooks and prediction application.

## Run the prediction application

From the project root, activate the environment and run:

```powershell
python main.py
```

The application requests all 12 features used by the trained pipeline. Enter whole numbers for area, bedrooms, bathrooms, stories, and parking; parking can be zero. Enter `yes` or `no` for the yes/no features and one of `furnished`, `semi-furnished`, or `unfurnished` for furnishing status. Blank or invalid answers are rejected and requested again.

The saved pipeline handles imputation and one-hot encoding. The application does not retrain the model or modify the dataset.

### Sample prediction

Example inputs:

```text
Area: 7420
Bedrooms: 4
Bathrooms: 2
Stories: 3
Main road access: yes
Guest room: no
Basement: no
Hot water heating: no
Parking spaces: 2
Air conditioning: yes
Preferred area: yes
Furnishing status: furnished
```

Example output from the saved model:

```text
Estimated house price: 7,968,276.13 (dataset price units)
```

The result is a model estimate, not a guaranteed sale price.

## Future improvements

- Add a user interface or web application.
- Compare Linear Regression with regularized and tree-based models.
- Use cross-validation and additional diagnostic checks.
- Add more location-specific data if a suitable licensed dataset is available.
- Improve input guidance and add automated tests for the command-line application.
