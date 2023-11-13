import pandas as pd
import numpy as np

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']

df = pd.read_csv(data_url, names=column_names, sep=r'\s*,\s*', engine='python')
ages = df['age'].tolist()



#Unary Coding LDP
def unary_coding_ldp(age, max_age, epsilon):
    unary_vector = [0] * (max_age + 1)
    unary_vector[age] = 1

    perturbed_vector = [0 if np.random.rand() > (np.exp(epsilon) / (1 + np.exp(epsilon))) 
                        else 1 for x in unary_vector]
    return perturbed_vector

def estimate_distribution_unary(perturbed_data, epsilon, max_age):
    N = len(perturbed_data)
    estimated_counts = [sum([user_data[age] for user_data in perturbed_data]) 
                        for age in range(max_age + 1)]
    estimated_distribution = [(count - N / (1 + np.exp(epsilon))) * 
                              (1 + np.exp(epsilon)) / np.exp(epsilon) 
                              for count in estimated_counts]
    return estimated_distribution

epsilon = 0.1
max_age = max(ages)
perturbed_data_unary = [unary_coding_ldp(age, max_age, epsilon) for age in ages]
estimated_distribution_unary = estimate_distribution_unary(perturbed_data_unary, epsilon, max_age)


#Generalized Random Response LDP
def generalized_random_response(age, max_age, epsilon):
    p = np.exp(epsilon) / (max_age + np.exp(epsilon))
    if np.random.rand() < p:
        return age
    else:
        return np.random.randint(0, max_age + 1)

def estimate_distribution_grr(perturbed_ages, epsilon, max_age):
    N = len(perturbed_ages)
    estimated_counts = [perturbed_ages.count(age) for age in range(max_age + 1)]
    estimated_distribution = [(count - N * 1 / (max_age + 1)) / 
                              (np.exp(epsilon) / (max_age + np.exp(epsilon))) 
                              for count in estimated_counts]
    return estimated_distribution

perturbed_ages_grr = [generalized_random_response(age, max_age, epsilon) for age in ages]
estimated_distribution_grr = estimate_distribution_grr(perturbed_ages_grr, epsilon, max_age)
