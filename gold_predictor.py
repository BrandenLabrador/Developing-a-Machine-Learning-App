"""
Gold Price Predictor
Entry-level machine learning application for commodity price prediction.

This version uses a supplied CSV file instead of downloading data automatically.
The CSV should be named gold_data.csv and should contain two columns:
Date, Price

Dataset format based on the DataHub Gold Prices dataset:
https://datahub.io/core/gold-prices
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


CSV_FILE = "gold_data.csv"
CHART_FILE = "gold_prediction_chart.png"


def load_data():
    """Load gold price data from the local CSV file."""
    data = pd.read_csv(CSV_FILE)

    print("First five rows of the dataset:")
    print(data.head())

    return data


def clean_data(data):
    """Clean the dataset and create simple date-based features."""
    data = data.dropna()

    # Convert the Date column to a real date.
    data["Date"] = pd.to_datetime(data["Date"])

    # Convert Price to a number in case the CSV reads it as text.
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.dropna()

    # Create simple features from the date.
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month

    # Time_Index gives each row a simple number order from oldest to newest.
    data = data.sort_values("Date")
    data["Time_Index"] = range(len(data))

    return data


def train_model(data):
    """Train a Random Forest model to predict gold price."""
    features = data[["Year", "Month", "Time_Index"]]
    target = data["Price"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    return model, x_test, y_test, predictions


def show_results(y_test, predictions):
    """Print the model evaluation results."""
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel Results")
    print("Mean Absolute Error:", round(mae, 2))
    print("R-squared Score:", round(r2, 2))


def predict_future_price(model, data):
    """Predict the price for the next month after the dataset ends."""
    last_date = data["Date"].max()
    next_date = last_date + pd.DateOffset(months=1)
    next_time_index = data["Time_Index"].max() + 1

    future_data = pd.DataFrame({
        "Year": [next_date.year],
        "Month": [next_date.month],
        "Time_Index": [next_time_index],
    })

    predicted_price = model.predict(future_data)[0]

    print("\nFuture Gold Price Prediction")
    print("Prediction Month:", next_date.strftime("%Y-%m"))
    print("Predicted Price: $" + str(round(predicted_price, 2)))


def create_chart(data, model):
    """Create a simple chart of actual and predicted gold prices."""
    features = data[["Year", "Month", "Time_Index"]]
    data["Predicted_Price"] = model.predict(features)

    plt.figure(figsize=(10, 5))
    plt.plot(data["Date"], data["Price"], label="Actual Price")
    plt.plot(data["Date"], data["Predicted_Price"], label="Predicted Price")
    plt.title("Actual vs Predicted Gold Prices")
    plt.xlabel("Date")
    plt.ylabel("Gold Price in USD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_FILE)
    print("\nChart saved as", CHART_FILE)
    plt.show()


def main():
    """Run the full machine learning application."""
    data = load_data()
    clean_gold_data = clean_data(data)

    model, x_test, y_test, predictions = train_model(clean_gold_data)

    show_results(y_test, predictions)
    predict_future_price(model, clean_gold_data)
    create_chart(clean_gold_data, model)


if __name__ == "__main__":
    main()
