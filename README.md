# RAG Context Selection with Contextual Influence Value
Context selection is a technique designed to enhance RAG (Retrieval-Augmented Generation) performance by selectively retaining high-quality retrieved contexts while filtering out low-quality ones, thereby shielding the LLM generator from noisy and irrelevant information. The Contextual Influence (CI) value represents an innovative context quality metric that comprehensively considers three key aspects: query-awareness, list-awareness, and generator-awareness. Another advantage of context selection using CI values is its simplicity in configuration - it only requires preserving contexts with positive CI values, eliminating the need for task-specific hyperparameter tuning such as top-k (the number of selected contexts).

The code is based on the [FlashRAG benchmark](https://github.com/RUC-NLPIR/FlashRAG).

## Quick Start
### Get Data and LLM Checkpoints
Install requirements:
```bash
pip install -r requirements.txt
```

Assign the `root_dir`
```bash
root_dir=path_to_the_root_dir
```

Download datasets from [FlashRAG datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) and save them to `root_dir/data/csm/base`.
Download the retrieval corpus form [wiki_corpus](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/tree/main/retrieval-corpus) and save the `.jsonl` file to `root_dir/indexs/wiki_index`.

Construct index with e5 dense retriver following [FlashRAG](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)  and save the `.index` file to `root_dir/indexs/wiki_index`.

Download LLM checkpoint folder and save it to `root_dir/ckpt`.

Here is an example of the folder structure of `root_dir`.
```bash
root_dir
|-- data
    |-- csm
        |-- base
        |-- refiner_scores
        |-- retrieval_cache
|-- ckpt
    |-- Llama-3.1-8B-Instruct
|-- indexs
    |-- wiki_index
        |-- e5_flat_inner.index
        |-- wiki18_100w.jsonl
|-- outputs
```

Specify tasks and running environments:
```bash
data_name=nq
mode=test
gpu_id=0
export PYTHONPATH=absolute_path_to_CSM_folder:$PYTHONPATH
```

### Cache Retrieval Results
We cache the retrieval results in `root_dir/csm/retrieval_cache` to avoid repeated calls to retriever. The number of retrieved contexts is set to `10` by default.
```bash
python scripts/retrieval_cache.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id
```

### Run!
```bash
python scripts/naive_llm.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id
python scripts/standard_rag.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id
python scripts/context_selection_rag.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id
```