# This module contains stuffs needed to analyze data. such as calculations
import numpy as np

from ChromaClientWrapper import getAllIds, getEmbeddingById


def calculateCosineSimilarity(a, b):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    a (array-like): The first vector.
    b (array-like): The second vector.

    Returns:
    float: The cosine similarity value.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculateSimilarityMatrix(collection):
    """
    A function to calculate a similarity matrix based on cosine similarity
    for a given collection of items.

    :param collection: The collection of items to calculate similarity for.
    :return: A dictionary representing the similarity matrix.
    """
    ids = getAllIds(collection)

    # Not 2d array, but a dictionary of dictionaries
    # Similarly to an adjacency matrix
    # So instead of getting an entry by [i][j] we get [idFirst][idSecond]
    similarityMatrix = {}
    for idFirst in ids:
        similarityMatrix[idFirst] = {}
        for idSecond in ids:
            similarityMatrix[idFirst][idSecond] = calculateCosineSimilarity(
                getEmbeddingById(collection, idFirst)["embeddings"][0],
                getEmbeddingById(collection, idSecond)["embeddings"][0])
    return similarityMatrix


def calculateCosineSimilarityForEntry(collection, idFirst, idSecond):
    """
    Calculate the cosine similarity between two entries in a collection.

    Parameters:
    - collection: The collection containing the entries.
    - idFirst: The ID of the first entry.
    - idSecond: The ID of the second entry.

    Returns:
    The cosine similarity between the embeddings of the two entries.
    """
    return calculateCosineSimilarity(getEmbeddingById(collection, idFirst)["embeddings"][0],
                                     getEmbeddingById(collection, idSecond)["embeddings"][0])
