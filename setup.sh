#!/bin/bash

mkdir -p src/rag_hub/{loaders,chunking,embeddings,vectorstore,retrievers,query,routing,generation,agents,tools,eval,multimodal}
mkdir -p notebooks scripts app eval_results docs
mkdir -p data/{raw,processed,eval}

for dir in src/rag_hub/*; do
  if [ -d "$dir" ]; then
    cat > "$dir/README.md" <<EOF
# $(basename "$dir")

This module is part of the RAG pipeline.

## Purpose
TODO: Describe this module.

EOF
  fi
done

touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/eval/.gitkeep

echo "RAG structure created!"