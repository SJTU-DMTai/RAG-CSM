root_dir=path_to_the_root_dir
data_name=nq
mode=test
gpu_id=3
export PYTHONPATH=absolute_path_to_CSM_folder:$PYTHONPATH


python scripts/retrieval_cache.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id
python scripts/naive_llm.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id
python scripts/standard_rag.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id
python scripts/context_selection_rag.py --root_dir $root_dir --data_name $data_name --mode $mode --gpu_id $gpu_id