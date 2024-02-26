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

if __name__ == '__main__':
    # mbpp = loadMbpp()
    mbpp = loadMbppOffline()
    print(mbpp)
