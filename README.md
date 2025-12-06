# RAG-CSM
This repo is the implementation of NeruIPS 2025 paper "[Influence Guided Context Selection for Effective Retrieval-Augmented Generation](https://openreview.net/pdf?id=ugaepulZyA)". This work proposes the Contextual Influence (CI) value for effective context selection, which comprehensively considers four key aspects: query-awareness, list-awareness, generator-awareness and configurarion-free. Since computing CI value is infeasible during inference, we propose CI surrogate model (CSM) for context selection.

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

Download CSM checkpoint folder from [google drive](https://drive.google.com/drive/folders/1AlO8olMDF7dTkGdtd8Uv3Hocd-WGSNnZ?usp=sharing) and save it to `root_dir/ckpt/csm`.

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
    |-- csm
        |-- nq_llama
            |-- model.safetensors
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
gpu_id='0'
export PYTHONPATH=absolute_path_to_RAG-CSM_folder:$PYTHONPATH
```

### Cache Retrieval Results
We cache the retrieval results in `root_dir/csm/retrieval_cache` to avoid repeated calls to retriever. The number of retrieved contexts is set to `10` by default.
```bash
python scripts/retrieval_cache.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id
```

### Run The Scripts
```bash
python scripts/naive_llm.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id
python scripts/standard_rag.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id
python scripts/context_selection_rag.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id --refiner ci
python scripts/context_selection_rag.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id --refiner csm
```

## Cite
If you find our code helpful, please consider citing our paper:
```text
@inproceedings{denginfluence,
  title={Influence Guided Context Selection for Effective Retrieval-Augmented Generation},
  author={Deng, Jiale and Shen, Yanyan and Pei, Ziyuan and Chen, Youmin and Huang, Linpeng},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```