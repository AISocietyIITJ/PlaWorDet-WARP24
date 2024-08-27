import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_predicted_vs_actual(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Plot the relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Predicted Price', y='Actual Price', data=df, color='blue', alpha=0.6)

    # Add a line of best fit
    sns.regplot(x='Predicted Price', y='Actual Price', data=df, scatter=False, color='red', line_kws={"lw": 2})

    # Set plot labels and title
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Relationship between Predicted and Actual Prices')

    # Show the plot
    plt.show()


def plot_predicted_vs_actual_with_similarity(file_path):
    df = pd.read_csv(file_path)
    # Plot the relationship with similarity score as hue
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        x='Predicted Price',
        y='Actual Price',
        hue='Similarity Score',
        data=df,
        palette='viridis',
        alpha=0.7
    )

    # Add a line of best fit
    sns.regplot(
        x='Predicted Price',
        y='Actual Price',
        data=df,
        scatter=False,
        color='red',
        line_kws={"lw": 2}
    )

    # Set plot labels and title
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Relationship between Predicted and Actual Prices with Similarity Score')

    # Add a color bar for the similarity score
    cbar = plt.colorbar(scatter.collections[0])
    cbar.set_label('Similarity Score')

    # Show the plot
    plt.show()


def load_and_describe_dataset(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Generate the statistical summary
    summary = df.describe()

    return summary


if __name__ == '__main__':
    file_path = './Data/ProcessedData.csv'
    summary = load_and_describe_dataset(file_path)
    plot_predicted_vs_actual(file_path)
    plot_predicted_vs_actual_with_similarity(file_path)
    print(summary)
