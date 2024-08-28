# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)


# %% [markdown]
# Q4:Dicrete input Discrete output

# %%
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)

# create fake data
def create_fake_data(N, M):
    X = pd.DataFrame(np.random.randint(2, size=(N, M)))
    y = pd.Series(np.random.randint(2, size=N))
    return X, y

# Function to measure time for learning and predicting
def measure_time(tree_type, X_train, y_train, X_test, num_runs=5):
    learning_times = []
    predict_times = []

    for _ in range(num_runs):
        tree = DecisionTree(criterion="information_gain", max_depth=5)

        # Measure time for learning the tree
        start_time = time.time()
        tree.fit(X_train, y_train)
        learning_time = time.time() - start_time

        # Measure time for predicting on test data
        start_time = time.time()
        y_pred = tree.predict(X_test)
        predict_time = time.time() - start_time

        learning_times.append(learning_time)
        predict_times.append(predict_time)

    # Calculate average and standard deviation
    avg_learning_time = np.mean(learning_times)
    std_learning_time = np.std(learning_times)
    avg_predict_time = np.mean(predict_times)
    std_predict_time = np.std(predict_times)

    return avg_learning_time, std_learning_time, avg_predict_time, std_predict_time

# Vary N and M
N_values = [100, 500, 1000]
M_values = [5, 10, 20]

# Lists to store average learning and predicting times, and std for each M
avg_learning_times_all = []
std_learning_times_all = []
avg_predict_times_all = []
std_predict_times_all = []

# Perform experiments and plot results
for M in M_values:
    avg_learning_times = []
    std_learning_times = []
    avg_predict_times = []
    std_predict_times = []

    for N in N_values:
        X, y = create_fake_data(N, M)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        avg_learning_time, std_learning_time, avg_predict_time, std_predict_time = measure_time("DecisionTree", X_train, y_train, X_test)
        avg_learning_times.append(avg_learning_time)
        std_learning_times.append(std_learning_time)
        avg_predict_times.append(avg_predict_time)
        std_predict_times.append(std_predict_time)

    # Append to the lists for all M values
    avg_learning_times_all.append(avg_learning_times)
    std_learning_times_all.append(std_learning_times)
    avg_predict_times_all.append(avg_predict_times)
    std_predict_times_all.append(std_predict_times)

# Plot average learning times with standard deviations for all M values
plt.figure()
for i, M in enumerate(M_values):
    plt.errorbar(N_values, avg_learning_times_all[i], yerr=std_learning_times_all[i], label=f'Learning Time (M={M})', marker='o')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Time (seconds)')
plt.title('Average Learning Time for Decision Tree')
plt.legend()

# Plot average predicting times with standard deviations for all M values
plt.figure()
for i, M in enumerate(M_values):
    plt.errorbar(N_values, avg_predict_times_all[i], yerr=std_predict_times_all[i], label=f'Predict Time (M={M})', marker='o')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Time (seconds)')
plt.title('Average Predicting Time for Decision Tree')
plt.legend()

plt.show()


# %% [markdown]
# Q4: Discrete inut Real Output

# %%
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Function to generate fake data
def generate_fake_data(N, M):
    data = {
        f"feature_{i}": np.random.randint(0, 2, N) for i in range(M)
    }
    data["target"] = np.random.randint(0, 2, N)
    return pd.DataFrame(data)

# Function to conduct experiments
def run_experiments(max_N, max_M):
    learning_times = []
    prediction_times = []

    for N in range(100, max_N + 1, 100):
        for M in range(2, max_M + 1):

            # Generate fake data
            fake_data = generate_fake_data(N, M)

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                fake_data.drop("target", axis=1), fake_data["target"], test_size=0.2, random_state=42
            )

            # Measure time for learning
            start_time = time.time()
            tree = DecisionTree(criterion='gini_index', max_depth=4)
            tree.fit(X_train, y_train)
            learning_time = time.time() - start_time

            # Measure time for prediction
            start_time = time.time()
            _ = tree.predict(X_test)
            prediction_time = time.time() - start_time

            learning_times.append((N, M, learning_time))
            prediction_times.append((N, M, prediction_time))

    return learning_times, prediction_times

# Main function to run experiments and plot results
def main():
    max_N = 1000
    max_M = 10

    learning_times, prediction_times = run_experiments(max_N, max_M)

    # Calculate average and std for learning times for each M
    avg_learning_times = {}
    std_learning_times = {}
    for M in range(2, max_M + 1):
        data = [time for _, _, time in learning_times if _ == M]
        avg_learning_times[M] = np.mean(data)
        std_learning_times[M] = np.std(data)

    # Calculate average and std for prediction times for each M
    avg_prediction_times = {}
    std_prediction_times = {}
    for M in range(2, max_M + 1):
        data = [time for _, _, time in prediction_times if _ == M]
        avg_prediction_times[M] = np.mean(data)
        std_prediction_times[M] = np.std(data)

    # Plotting learning times with error bars
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for M in range(2, max_M + 1):
        data = [(N, time) for N, _, time in learning_times if _ == M]
        avg = avg_learning_times[M]
        std = std_learning_times[M]
        plt.errorbar(*zip(*data), yerr=std, label=f"M={M}, Avg={avg:.2f}")
    plt.title("Learning Time vs. N for Different M")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Learning Time (s)")
    plt.legend()

    # Plotting prediction times with error bars
    plt.subplot(1, 2, 2)
    for M in range(2, max_M + 1):
        data = [(N, time) for N, _, time in prediction_times if _ == M]
        avg = avg_prediction_times[M]
        std = std_prediction_times[M]
        plt.errorbar(*zip(*data), yerr=std, label=f"M={M}, Avg={avg:.2f}")
    plt.title("Prediction Time vs. N for Different M")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Prediction Time (s)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


# %% [markdown]
# Q4: Real input Discrete Output

# %%
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Experiment parameters
Ns = [100, 500, 1000]  # Vary the number of samples
Ms = [5, 10, 20]  # Vary the number of real-valued features

fit_times_discrete = []
prediction_times_discrete = []

# Perform experiments
for N in Ns:
    for M in Ms:
        fit_times_per_experiment = []
        prediction_times_per_experiment = []

        # Generate fake data
        X_fake = np.random.rand(N, M)  # Real-valued features
        y_fake = np.random.choice([0, 1], size=N)  # Discrete output with 2 classes

        X_fake_train, X_fake_test, y_fake_train, _ = train_test_split(X_fake, y_fake, test_size=0.3, random_state=42)

        # Convert to DataFrame for real input features
        X_fake_train = pd.DataFrame(X_fake_train, columns=[str(i) for i in range(M)])
        X_fake_test_df = pd.DataFrame(X_fake_test, columns=[str(i) for i in range(M)])
        y_fake_train = pd.Series(y_fake_train)

        # Measure fit time for discrete output
        start_time = time.time()
        tree = DecisionTree(criterion='gini_index', max_depth=4)
        tree.fit(X_fake_train, y_fake_train)
        end_time = time.time()
        fit_time_discrete = end_time - start_time
        fit_times_per_experiment.append(fit_time_discrete)

        # Measure prediction time for discrete output
        start_time = time.time()
        y_pred = tree.predict(X_fake_test_df)
        end_time = time.time()
        prediction_time_discrete = end_time - start_time
        prediction_times_per_experiment.append(prediction_time_discrete)

        # Calculate average and std for each experiment
        avg_fit_time = np.mean(fit_times_per_experiment)
        std_fit_time = np.std(fit_times_per_experiment)
        fit_times_discrete.append((avg_fit_time, std_fit_time))

        avg_prediction_time = np.mean(prediction_times_per_experiment)
        std_prediction_time = np.std(prediction_times_per_experiment)
        prediction_times_discrete.append((avg_prediction_time, std_prediction_time))

# Plot the results for discrete output with error bars
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
avg_fit_times, std_fit_times = zip(*fit_times_discrete)
plt.errorbar(range(len(avg_fit_times)), avg_fit_times, yerr=std_fit_times, marker='o')
plt.xticks(range(len(avg_fit_times)), [f"N={N}, M={M}" for N in Ns for M in Ms], rotation=45)
plt.title('Decision Tree Learning Time (Discrete Output)')
plt.xlabel('Experiment')
plt.ylabel('Time (seconds)')

plt.subplot(1, 2, 2)
avg_prediction_times, std_prediction_times = zip(*prediction_times_discrete)
plt.errorbar(range(len(avg_prediction_times)), avg_prediction_times, yerr=std_prediction_times, marker='o', color='orange')
plt.xticks(range(len(avg_prediction_times)), [f"N={N}, M={M}" for N in Ns for M in Ms], rotation=45)
plt.title('Decision Tree Prediction Time (Discrete Output)')
plt.xlabel('Experiment')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.show()


# %% [markdown]
# Q4: real input real output

# %%
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data
def create_fake_data(N, P):
    X = pd.DataFrame(np.random.rand(N, P), columns=[f'feature_{i}' for i in range(P)])
    y = pd.Series(np.random.randint(2, size=N), name='target')
    return X, y


# Function to calculate average time taken by fit() and predict()
def calculate_average_times(N_values, P_values, criterion, max_depth):
    results = {"N": [], "P": [], "Fit Time": [], "Predict Time": []}

    for N in N_values:
        for P in P_values:
            fit_times = []
            predict_times = []

            for _ in range(num_average_time):
                X, y = create_fake_data(N, P)

                decision_tree = DecisionTree(criterion=criterion, max_depth=max_depth)

                # Measure fit time
                start_time = time.time()
                decision_tree.fit(X, y)
                fit_time = time.time() - start_time
                fit_times.append(fit_time)

                # Measure predict time
                start_time = time.time()
                _ = decision_tree.predict(X)
                predict_time = time.time() - start_time
                predict_times.append(predict_time)

            avg_fit_time = np.mean(fit_times)
            avg_predict_time = np.mean(predict_times)

            results["N"].append(N)
            results["P"].append(P)
            results["Fit Time"].append(avg_fit_time)
            results["Predict Time"].append(avg_predict_time)

    return results


# Function to plot the results
def plot_results(results, criterion):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(results["N"], results["Fit Time"], label="Fit Time")
    plt.title(f"{criterion} - Average Fit Time vs N")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Average Fit Time (seconds)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(results["P"], results["Predict Time"], label="Predict Time")
    plt.title(f"{criterion} - Average Predict Time vs P")
    plt.xlabel("Number of Features (P)")
    plt.ylabel("Average Predict Time (seconds)")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Run the functions, Learn the DTs, and Show the results/plots
def run_experiments():
    N_values = [100, 500, 1000]
    P_values = [10, 50, 100]
    criterion = "information_gain"
    max_depth = 5

    results = calculate_average_times(N_values, P_values, criterion, max_depth)
    plot_results(results, criterion)


if __name__ == "__main__":
    run_experiments()


# %% [markdown]
# 


