import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Loading
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']

df = pd.read_csv(data_url, names=column_names, sep=r'\s*,\s*', engine='python')
ages = df['age'].tolist()

# LDP Protocols

def unary_coding(data, epsilon):
    perturbed_data = []
    for age in data:
        p = np.exp(epsilon) / (np.exp(epsilon) + len(set(data)) - 1)
        if np.random.rand() < p:
            perturbed_data.append(age)
        else:
            perturbed_data.append(np.random.choice(list(set(data))))
    return perturbed_data

def generalized_random_response(data, epsilon):
    perturbed_data = []
    p = np.exp(epsilon) / (np.exp(epsilon) + len(set(data)) - 1)
    for age in data:
        if np.random.rand() < p:
            perturbed_data.append(age)
        else:
            perturbed_data.append(np.random.choice(list(set(data))))
    return perturbed_data

def frequency_estimation(perturbed_data, epsilon, true_data):
    estimated_counts = {}
    for age in set(true_data):
        observed_count = perturbed_data.count(age)
        p = np.exp(epsilon) / (np.exp(epsilon) + len(set(true_data)) - 1)
        estimated_count = (observed_count - (1-p) * len(true_data) / len(set(true_data))) / p
        estimated_counts[age] = estimated_count
    return estimated_counts

# Simulation of LDP protocols

epsilons = range(1, 11)
l1_distances_unary = []
l1_distances_grr = []

for epsilon in epsilons:
    perturbed_data_unary = unary_coding(ages, epsilon)
    estimated_counts_unary = frequency_estimation(perturbed_data_unary, epsilon, ages)
    
    perturbed_data_grr = generalized_random_response(ages, epsilon)
    estimated_counts_grr = frequency_estimation(perturbed_data_grr, epsilon, ages)
    
    l1_distance_unary = sum(abs(estimated_counts_unary[age] - ages.count(age)) for age in set(ages))
    l1_distances_unary.append(l1_distance_unary)
    
    l1_distance_grr = sum(abs(estimated_counts_grr[age] - ages.count(age)) for age in set(ages))
    l1_distances_grr.append(l1_distance_grr)

# Plotting

plt.figure(figsize=(10, 6))
plt.plot(epsilons, l1_distances_unary, label='Unary Coding', marker='o')
plt.plot(epsilons, l1_distances_grr, label='Generalized Random Response', marker='x')
plt.xlabel('Epsilon (ε)')
plt.ylabel('L1-distance')
plt.title('Comparison of LDP protocols with different ε values')
plt.legend()
plt.grid(True)
plt.savefig("q2-b.png", dpi=300, bbox_inches='tight')
print("done")
# plt.show()
