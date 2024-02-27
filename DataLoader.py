from ChromaClientWrapper import initChromaClient, initCollection, upsertEntry, addEntry
from CollectionEntry import CollectionEntry
from datasets import load_dataset

def loadMbpp(isFullSubset=False):
    return load_dataset("mbpp") if isFullSubset else load_dataset("mbpp", "sanitized")

def loadMbppOffline():
    data_files = {
        "prompt": "prompt-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
        "train": "train-00000-of-00001.parquet",
        "validation": "validation-00000-of-00001.parquet"
    }
    return load_dataset("parquet",
                        data_dir="datasets/mbpp_sanitized/",
                        data_files=data_files)

def loadMbppToChroma(splits=None):
    if splits is None:
        splits = ["train", "test"]
    client = initChromaClient()
    collection = initCollection(client, "mbpp-sanitized")

    mbpp_dataset = loadMbppOffline()
    for split in splits:
        loadSplit(collection, mbpp_dataset, split, "prompt", "task_id")
    return client, collection
def loadSplit(collection, dataset, split, documentField, idField):
    sanitizeMetadataInDataset(dataset)
    for row in dataset[split]:
        # collection.upsert(
        #     documents=row[documentField],
        #     metadatas=row,
        #     ids=row[idField]
        # )
        upsertEntry(collection, CollectionEntry(row[documentField], row, str(row[idField])))
        # addEntry(collection, CollectionEntry(row[documentField], row, idField))
        print(collection.count())


#Remove everything that is not a str, int, float or bool
def sanitizeMetadataInDataset(dataset):
    for split in dataset:
        for row in dataset[split]:
            sanitizeMetadata(row)


def sanitizeMetadata(metadata):
    sanitizedMetadata = metadata.copy()
    for key, value in metadata.items():
        if isinstance(value, bool) or isinstance(value, (str, int, float)):
            sanitizedMetadata[key] = value
    return sanitizedMetadata


if __name__ == '__main__':
    # mbpp = loadMbpp()
    mbpp = loadMbppOffline()
    print("Length of train", len(mbpp["train"]))
    print("Length of test", len(mbpp["test"]))

    client, collection = loadMbppToChroma()
    # client = initChromaClient()
    # collection = initCollection(client, "mbpp-sanitized")
    # testProblemQuery = "Write a python function to find the first repeated character in a given string."

    print(collection.count())
    # print(collection.get(
    #     ids=["602"]
    # ))

    # collection.query(
    #     query_texts=[testProblemQuery],
    #     n_results = 1,
    #     where={
    #         "task_id": {
    #             "$ne": 602
    #         }
    #     }
    # )


    # print(mbpp["train"][0])
