import pandas as pd
import numpy as np
from math import sqrt, pi, exp
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

encoder = LabelEncoder()
dataset['class'] = encoder.fit_transform(dataset['class'])

# Split the dataset into training and test
test_data = dataset.iloc[list(range(0,10)) + list(range(50,60)) + list(range(100,110))]
train_data = dataset.drop(test_data.index)

# Training function
def train_naive_bayes(train_data):
    classes = train_data['class'].unique()
    class_data = {}
    for c in classes:
        data = train_data[train_data['class'] == c]
        class_data[c] = {
            'prior_prob': len(data) / len(train_data),
            'summary': {
                'mean': data.mean(),
                'std': data.std()
            }
        }
    return class_data

# Calculate Gaussian Probability Density Function
def gaussian_pdf(x, mean, std):
    exponent = exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (sqrt(2 * pi) * std)) * exponent

# Predict function
def predict(model, test_data):
    predictions = []
    for _, row in test_data.iterrows():
        probs = {}
        for c, data in model.items():
            probs[c] = data['prior_prob']
            for column, value in row.items():
                if column != 'class':
                    probs[c] *= gaussian_pdf(value, data['summary']['mean'][column], data['summary']['std'][column])
        predictions.append(max(probs, key=probs.get))
    return predictions

# Training the model
model = train_naive_bayes(train_data)

# Predicting for test data
predictions = predict(model, test_data)

# Display the results
results = pd.concat([test_data['class'].reset_index(drop=True), pd.Series(predictions, name='Predicted')], axis=1)
print(results)
# Convert DataFrame to image
fig, ax = plt.subplots(figsize=(10, 4)) # set the size that you'd like (width, height)
ax.axis('off')
tbl = ax.table(cellText=results.values, colLabels=results.columns, cellLoc = 'center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.2, 1.2)
plt.savefig("q1-results.png", dpi=300, bbox_inches='tight')
plt.show()
