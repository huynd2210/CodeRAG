from CollectionEntry import CollectionEntry
import chromadb
from chromadb.utils import embedding_functions

#ChromaDB for some reason doesnt work with class as wrapper (No OOP).


def addEntry(collection, collectionEntry: CollectionEntry):
    collection.add(
        documents=collectionEntry.document,
        metadatas=[collectionEntry.metadata],
        ids=[collectionEntry.id]
    )

    #For whatever reason chromaDB doesnt save embeddings in the embeddings field. The embeddings exists, but must be retrieved separately
def isMetadataPrimitive(metadata):
    return isinstance(metadata, (str, int, float, bool))

def upsertEntry(collection, collectionEntry: CollectionEntry):
    #Remove all metadata that is not a str, int, float or bool
    collectionEntry.metadata = {k: v for k, v in collectionEntry.metadata.items() if isMetadataPrimitive(v)}

    collection.upsert(
        documents=collectionEntry.document,
        metadatas=[collectionEntry.metadata],
        ids=[collectionEntry.id]
    )

    #For whatever reason chromaDB doesnt save embeddings in the embeddings field. The embeddings exists, but must be retrieved separately
    # collection.upsert(
    #     documents=collectionEntry.document,
    #     metadatas=[collectionEntry.metadata],
    #     embeddings=getEmbeddingById(collection, collectionEntry.id)["embeddings"],
    #     ids=[collectionEntry.id]
    # )

def query(collection, queryTexts, topK=10, where=None, whereDocuments=None):
    return collection.query(
        query_texts=queryTexts,
        where=where,
        where_documents=whereDocuments,
        n_results=topK
    )


def getAllEmbeddings(collection):
    return collection.get(include=["embeddings"])

def getAllIds(collection):
    return collection.get(include=["ids"])

def getEntryById(collection, id):
    return collection.get(ids=[id])

def getEmbeddingById(collection, id):
    return collection.get(ids=[id], include=["embeddings"])

def initCollection(client, collectionName, collectionMetadata=None, embeddingModel="all-MiniLM-L6-v2"):
    """
    Initialize a new collection in the database with the specified name and optional metadata.

    Parameters:
        client (Client): The client object for connecting to the database.
        collectionName (str): The name of the collection to be created.
        collectionMetadata (dict, optional): Additional metadata for the collection. Defaults to None.
        embeddingModel (str): The embedding model to be used. Defaults to "all-MiniLM-L6-v2".

    Returns:
        Collection: The newly created or existing collection object.
    """
    collectionMetadata = {"hnsw:space": "cosine"} if collectionMetadata is None else collectionMetadata

    sentenceTransformerEmbeddingFunction = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embeddingModel)

    embeddingFunction = sentenceTransformerEmbeddingFunction

    collection = client.get_or_create_collection(name=collectionName,
                                                 embedding_function=embeddingFunction,
                                                 metadata=collectionMetadata)
    return collection


def initChromaClient(path="./data"):
    """
    Initializes a Chroma client with the specified path.

    :param path: The path to the data directory (default="./data").
    :return: A PersistentClient object.
    """
    client = chromadb.PersistentClient(path=path)
    return client

def initChroma(collectionName):
    """
    Initializes the Chroma client and a specific collection.

    Parameters:
        collectionName (str): The name of the collection to initialize (get or create).

    Returns:
        tuple: A tuple containing the client and the initialized collection.
    """
    client = initChromaClient()
    collection = initCollection(client, collectionName)
    return client, collection

# test
if __name__ == '__main__':
    client = initChromaClient()
    collection = initCollection(client, "problemstore")

    testProblem = """
      We need a simple function that determines if a plural is needed or not. It should take a number, and return true if a plural should be used with that number or false if not. This would be useful when printing out a string such as 5 minutes, 14 apples, or 1 sun.

      You only need to worry about english grammar rules for this kata, where anything that isn't singular (one of something), it is plural (not one of something).

      All values will be positive integers or floats, or zero.
      """

    testProblem2 = """
    Write python code to find the factorial of a number
    """

    collectionEntry = CollectionEntry(testProblem, {"test": "test"}, id_generatingStrategy="md5")
    collectionEntry2 = CollectionEntry(testProblem2, {"test": "test"}, id_generatingStrategy="md5")

    # addEntry(collection, collectionEntry)
    upsertEntry(collection, collectionEntry)
    upsertEntry(collection, collectionEntry2)
    print(collection.count())

    allEmbeddings = getAllEmbeddings(collection)


    client.delete_collection("problemstore")
