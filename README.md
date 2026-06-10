# Gold Price Predictor

This is a Python machine learning app that predicts gold prices using a supplied CSV file.

## Files

- `gold_predictor.py` - Python source code
- `gold_data.csv` - supplied CSV dataset in DataHub format
- `requirements.txt` - required Python libraries
- `gold_prediction_chart.png` - generated chart after running the app
- `Assignment 1- Developing a Machine Learning App.pdf` - project report write up

## Dataset

The project uses a CSV file in the same format as the DataHub Gold Prices dataset:

https://datahub.io/core/gold-prices

The CSV should contain:

```text
Date,Price
2018-01,1331.30
2018-02,1330.73
```

## How to Run

1. Install Python 3.
2. Open the project folder in VS Code or Terminal.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python3 gold_predictor.py
```

5. Review the terminal output and the generated chart.
