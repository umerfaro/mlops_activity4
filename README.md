# Wine Dataset RandomForest with MLflow

This project trains a RandomForestClassifier on the Wine dataset and logs metrics, parameters, and the model to MLflow.

## Setup

1. Create a virtual environment:

```
python -m venv venv
```

2. Activate the virtual environment:

- On Windows:

```
venv\Scripts\activate
```

- On Unix or MacOS:

```
source venv/bin/activate
```

3. Install the required packages:

```
pip install -r requirements.txt
```

## Running the Code

To train the model and log to MLflow:

```
python train_wine_rf.py
```

## Viewing MLflow Results

Start the MLflow UI to view the experiment results:

```
mlflow ui
```

Then open a browser and navigate to http://localhost:5000 to view the experiment results.
