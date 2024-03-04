import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

def visualize_similarity_matrix(similarity_matrix: np.ndarray,
                                item_names: list,
                                colorMap: str = "coolwarm",
                                annot: bool = True,
                                fmt: str = ".2f",
                                figureSize: Tuple[int, int] = (8, 8),
                                xLabel: Optional[str] = None,
                                yLabel: Optional[str] = None,
                                title: Optional[str] = None,
                                ) -> None:
    """
    Visualize a similarity matrix using a heatmap.

    Parameters:
    - similarity_matrix (numpy.ndarray): The similarity matrix to be visualized.
    - item_names (list): List of item names corresponding to the rows and columns of the similarity matrix.
    - colorMap (str): Colormap for the heatmap (default is "coolwarm").
    - annot (bool): If True, display the numeric values in the heatmap cells (default is True).
    - fmt (str): String formatting code for annot values (default is ".2f").
    - figureSize (Tuple[int, int]): Size of the figure (default is (8, 8)).

    Returns:
    - None: Displays the heatmap.
    """

    # Set up the plot
    plt.figure(figsize=figureSize)
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Create a heatmap using Seaborn
    heatmap = sns.heatmap(similarity_matrix, cmap=colorMap, annot=annot, fmt=fmt, square=True, linewidths=.5)

    heatmap.set_xticklabels(item_names, rotation=45, ha="right")
    heatmap.set_yticklabels(item_names, rotation=0)

    # Add labels and title
    plt.xlabel("Items") if xLabel is None else plt.xlabel(xLabel)
    plt.ylabel("Items") if yLabel is None else plt.ylabel(yLabel)
    plt.title("Similarity Matrix") if title is None else plt.title(title)

    # Show the plot
    plt.show()


def main():
    # Example data
    num_items = 5
    item_names = [f"Item_{i}" for i in range(num_items)]

    # Create a random similarity matrix
    np.random.seed(42)
    similarity_matrix = np.random.rand(num_items, num_items)

    # Testing the function
    visualize_similarity_matrix(similarity_matrix, item_names)


if __name__ == '__main__':
    main()
