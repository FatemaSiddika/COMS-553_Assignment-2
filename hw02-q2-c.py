import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data Loading
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']

df = pd.read_csv(data_url, names=column_names, sep=r'\s*,\s*', engine='python')
all_ages = df['age'].tolist()

# Define LDP Protocols

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

# Simulation for varying numbers of users
epsilon = 2
percentages = [i/10 for i in range(1, 11)]  # 10% to 100% in increments of 10%
l1_distances_unary = []
l1_distances_grr = []

l1_distances_unary_relative = []
l1_distances_grr_relative = []

for percentage in percentages:
    num_samples = int(len(all_ages) * percentage)
    sampled_ages = np.random.choice(all_ages, num_samples, replace=False)
    
    perturbed_data_unary = unary_coding(sampled_ages, epsilon)
    estimated_counts_unary = frequency_estimation(perturbed_data_unary, epsilon, sampled_ages)
    
    perturbed_data_grr = generalized_random_response(sampled_ages, epsilon)
    estimated_counts_grr = frequency_estimation(perturbed_data_grr, epsilon, sampled_ages)
    
    #l1_distance_unary = sum(abs(estimated_counts_unary[age] - sampled_ages.count(age)) for age in set(sampled_ages))
    l1_distance_unary = sum(abs(estimated_counts_unary[age] - np.sum(sampled_ages == age)) for age in set(sampled_ages))
    l1_distances_unary.append(l1_distance_unary)
    
    #l1_distance_grr = sum(abs(estimated_counts_grr[age] - sampled_ages.count(age)) for age in set(sampled_ages))
    l1_distance_grr = sum(abs(estimated_counts_grr[age] - np.sum(sampled_ages == age)) for age in set(sampled_ages))
    l1_distances_grr.append(l1_distance_grr)

    total_count = len(sampled_ages)
    l1_distances_unary_relative.append(l1_distance_unary / total_count)
    l1_distances_grr_relative.append(l1_distance_grr / total_count)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(percentages, l1_distances_unary, label='Unary Coding', marker='o')
plt.plot(percentages, l1_distances_grr, label='Generalized Random Response', marker='x')
plt.xlabel('Percentage of Users')
plt.ylabel('L1-distance')
plt.title('Comparison of LDP protocols with varying user percentages and fixed ε=2')
plt.legend()
plt.grid(True)
plt.grid(True)
plt.savefig("q2-c-1.png", dpi=300, bbox_inches='tight')
#plt.show()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(percentages, l1_distances_unary_relative, label='Unary Coding - Relative', marker='o')
plt.plot(percentages, l1_distances_grr_relative, label='Generalized Random Response - Relative', marker='x')
plt.xlabel('Percentage of Users')
plt.ylabel('Relative L1-distance')
plt.title('Comparison of LDP protocols with varying user percentages and fixed ε=2')
plt.legend()
plt.grid(True)
plt.savefig("q2-c-2.png", dpi=300, bbox_inches='tight')
#plt.show()
