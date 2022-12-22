# Breast Cancer Detection: 
Basic ML script that detects individuals at risk for breast cancer

# Getting Started:
1. Clone the repo or download the zip to get started.
2. Open terminal and navigate to the project folder
```zsh
cd sentiment_analysis
```
3. Run requirements.txt file
```zsh
pip install -r requirements.txt
```

## Instructions:
1. To play around with model, you can edit the number range in the following code: 
```python
x_new = np.array(random.sample(range(0,50), len(data['feature_names'])))
```
2. If you have your own numbers you want to test against the model, you can change x_new to:
```python
x_new = np.array([your, numbers, go, here])
```
