# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_time_series(x_test_path, x_gen_path, output_dir):
    """
    Plots time series data for classes 3 and 7 from X_test, and random 40 time series from X_gen.
    Args:
    - x_test_path: Path to the TSV file containing X_test data with class labels.
    - x_gen_path: Path to the TSV file containing X_gen data without class labels.
    - output_dir: Directory where the plots will be saved.
    """
    # Load X_test from TSV
    x_test_data = pd.read_csv(x_test_path, sep='\t', header=None)
    y_test = x_test_data.iloc[:, 0]  # Class labels
    X_test = x_test_data.drop(columns=0).values  # Time series data

    # Filter for classes 3 and 7, selecting 20 time series each
    indices_class_3 = np.random.choice(np.where(y_test == 3)[0], 5, replace=False)
    indices_class_7 = np.random.choice(np.where(y_test == 7)[0], 5, replace=False)

    # Load X_gen from TSV
    X_gen = pd.read_csv(x_gen_path, sep='\t', header=None).values
    indices_x_gen = np.random.choice(range(X_gen.shape[0]), 10, replace=False)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot for classes 3 and 7
    for idx in np.concatenate([indices_class_3, indices_class_7]):
        axes[0].plot(X_test[idx], alpha=0.5)
    axes[0].set_title("Combined original data for Person 3 and Person 7")
    axes[0].set_xlabel("Time steps")
    axes[0].set_ylabel("Values")
    axes[0].grid(True)

    # Second subplot for X_gen
    for idx in indices_x_gen:
        axes[1].plot(X_gen[idx], alpha=0.5)
    axes[1].set_title("Generated data for a new individual using Person 3 and 7's data")
    axes[1].set_xlabel("Time steps")
    axes[1].set_ylabel("Value")
    axes[1].grid(True)

    plt.tight_layout()
    # Save the complete figure with both subplots
    fig.savefig(f"{output_dir}/combined_plots_inclinedown.png")
    plt.close(fig)

if __name__ == "__main__":
    # Define file paths and output directory
    x_test_path = 'C:/Users/USER/OneDrive/Desktop/NUS_Varun_TimeVQVAE/datasets/UCRArchive_2018/IMU_inclinedown/IMU_inclinedown_TEST.tsv'
    x_gen_path = 'C:/Users/USER/OneDrive/Desktop/NUS_Varun_TimeVQVAE/Results_inclinedown/New_person_Class_2_and_6.tsv'
    output_dir = 'C:/Users/USER/OneDrive/Desktop/NUS_Varun_TimeVQVAE/Plots'
    
    # Call the plotting function
    plot_time_series(x_test_path, x_gen_path, output_dir)
