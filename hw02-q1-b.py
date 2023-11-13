import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DifferentiallyPrivateNaiveBayes:
    
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.mu = {}
        self.sigma = {}
        self.class_probs = {}

    def laplace_noise(self, sensitivity, epsilon):
        scale = sensitivity / epsilon
        return np.random.laplace(0, scale)

    def fit(self, X, y):
        labels = set(y)
        
        for label in labels:
            data_for_label = X[y == label]
            
            # Using Laplace smoothing for class probabilities
            self.class_probs[label] = (len(data_for_label) + 1) / (len(X) + len(labels))
            
            self.mu[label] = {}
            self.sigma[label] = {}

            epsilon_mu = self.epsilon / (3*4)
            epsilon_sigma = self.epsilon / (3*4)

            for feature_idx in range(X.shape[1]):
                mu = np.mean(data_for_label[:, feature_idx])
                sigma = np.std(data_for_label[:, feature_idx])

                # Calculating sensitivity for mean and standard deviation
                s_mu = (max(data_for_label[:, feature_idx]) - min(data_for_label[:, feature_idx])) / (len(data_for_label) + 1)
                s_sigma = np.sqrt(len(data_for_label)) * (max(data_for_label[:, feature_idx]) - min(data_for_label[:, feature_idx])) / (len(data_for_label) + 1)
                
                # Adding Laplace noise with some smoothing
                self.mu[label][feature_idx] = mu + self.laplace_noise(s_mu, epsilon_mu) + 1e-3
                self.sigma[label][feature_idx] = sigma + self.laplace_noise(s_sigma, epsilon_sigma) + 1e-3
                

# Load dataset from the provided URL and encode the class labels
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

encoder = LabelEncoder()
dataset['class'] = encoder.fit_transform(dataset['class'])

data = dataset[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']].values
labels = dataset['class'].values

# Extracting training and testing data
test_indices = list(range(0, 10)) + list(range(50, 60)) + list(range(100, 110))
train_data = np.delete(data, test_indices, axis=0)
train_labels = np.delete(labels, test_indices, axis=0)

# Training the Differentially Private Naive Bayes model
model = DifferentiallyPrivateNaiveBayes(epsilon=1.0)
model.fit(train_data, train_labels)



# Print the model details
print("Differentially Private Naive Bayes Model Details:\n")

print("Class Probabilities:")
for class_label, prob in model.class_probs.items():
    print(f"Class {class_label}: {prob:.4f}")

print("\nMeans for each feature per class:")
for class_label, features in model.mu.items():
    print(f"Class {class_label}:")
    for feature_idx, mu_value in features.items():
        feature_name = names[feature_idx]
        print(f"  {feature_name}: {mu_value:.4f}")

print("\nStandard Deviations for each feature per class:")
for class_label, features in model.sigma.items():
    print(f"Class {class_label}:")
    for feature_idx, sigma_value in features.items():
        feature_name = names[feature_idx]
        print(f"  {feature_name}: {sigma_value:.4f}")
