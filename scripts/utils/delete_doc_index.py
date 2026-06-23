import argparse
from rag_hub.vectorstore.qdrant_store import QdrantStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_name", required=True, help="Document name to delete")
    args = parser.parse_args()

    store = QdrantStore(collection="financebench_v1")

    store.delete_by_doc_name(args.doc_name)


if __name__ == "__main__":
    main()