import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ChromaClientWrapper import initChromaClient, initCollection, addEntry, getAllEmbeddings
from CollectionEntry import CollectionEntry

# Example text embeddings (replace with your own)
texts = ["apple", "banana", "orange", "grape", "kiwi", "pear"]
 # Replace with your actual embeddings


# Function to plot embeddings using PCA or t-SNE
def visualize_embeddings(embeddings, method='PCA'):
    if method == 'PCA':
        reducer = PCA(n_components=2)
        reduced_embeddings = reducer.fit_transform(embeddings)
    elif method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Unsupported dimensionality reduction method")

    plt.figure(figsize=(8, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='b', marker='o')
    for i, text in enumerate(texts):
        plt.annotate(text, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

    plt.title(f'Text Embeddings Visualization using {method}')
    plt.show()

def visualize_embeddings_tsne(embeddings, perplexity=30):
    # Reshape the embeddings into a 2D array
    embeddings_array = np.array(embeddings).reshape(1, -1)

    # Initialize t-SNE model with specified perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)

    # Fit and transform the data
    embeddings_tsne = tsne.fit_transform(embeddings_array)

    # Plot the t-SNE visualization
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
    plt.title(f't-SNE Visualization of Embeddings (Perplexity={perplexity})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
# # Visualize using PCA
# visualize_embeddings(embeddings, method='PCA')
#
# # Visualize using t-SNE
# visualize_embeddings(embeddings, method='t-SNE')


def main():
    client = initChromaClient()
    collection = initCollection(client, "problemstore")

    testProblem = """
          We need a simple function that determines if a plural is needed or not. It should take a number, and return true if a plural should be used with that number or false if not. This would be useful when printing out a string such as 5 minutes, 14 apples, or 1 sun.

          You only need to worry about english grammar rules for this kata, where anything that isn't singular (one of something), it is plural (not one of something).

          All values will be positive integers or floats, or zero.
          """
    collectionEntry = CollectionEntry(testProblem, {"test": "test"}, id_generatingStrategy="md5")

    addEntry(collection, collectionEntry)

    allEmbeddings = getAllEmbeddings(collection)

    print(allEmbeddings['embeddings'][0])
    visualize_embeddings_tsne(allEmbeddings['embeddings'][0])
    # visualize_embeddings(np.array(allEmbeddings['embeddings'][0]), method='t-SNE')

    client.delete_collection("problemstore")


if __name__ == '__main__':
    main()
