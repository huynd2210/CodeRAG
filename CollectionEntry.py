import uuid
import hashlib

class CollectionEntry:
    def __init__(self, document, metadata, id=None, id_generatingStrategy=None):
        self.document = document
        self.metadata = metadata
        if id is None and id_generatingStrategy is not None:
            self.genereateId(id_generatingStrategy, document=document)
        else:
            self.id = id if id is not None else str(uuid.uuid4())

    def genereateId(self, id_generatingStrategy, document=None):
        if id_generatingStrategy == "uuid":
            self.id = str(uuid.uuid4())
        elif id_generatingStrategy == "md5":
            self.id = hashlib.md5(document.encode("utf-8")).hexdigest()
        elif id_generatingStrategy == "sha-1":
            self.id = hashlib.sha1(document.encode("utf-8")).hexdigest()
        elif id_generatingStrategy == "sha-256":
            self.id = hashlib.sha256(document.encode("utf-8")).hexdigest()
