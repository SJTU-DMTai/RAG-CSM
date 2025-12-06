from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.config import Config
import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--root_dir', type=str)
    args.add_argument('--data_name', type=str, default='nq')
    args.add_argument('--retriever', type=str, default='e5')
    args.add_argument('--gpu_id', type=int, default=1)
    args.add_argument('--retrieval_topk', type=int, default=10)
    args.add_argument('--mode', type=str, default='test')
    args = args.parse_args()

    data_name = args.data_name
    root_dir = args.root_dir
    model2path = {
        'e5': f'{root_dir}/ckpt/e5-base-v2',
        'llama3-8B-instruct': f'{root_dir}/ckpt/Llama-3.1-8B-Instruct',
        'qwen2.5-7b-instruct': f'{root_dir}/ckpt/Qwen2.5-7B-Instruct',
        'bert': f'{root_dir}/ckpt/bert-base-uncased',
    }

    config_dict = {
        'dataset_name': data_name,
        'split': [args.mode],
        'gpu_id': args.gpu_id,
        'data_dir': f'{root_dir}/data/csm/base',
        'save_dir': f'{root_dir}/outputs/',
        'model2path': model2path,
        'corpus_path': f'{root_dir}/index/wiki_index/wiki18_100w.jsonl',
        'index_path': f'{root_dir}/index/wiki_index/e5_flat_inner.index',
        # --- retriever ---
        'save_retrieval_cache': True,
        'use_retrieval_cache': False,
        'retrieval_cache_path': f'{root_dir}/data/csm/retrieval_cache/{data_name}_{args.mode}.json',
        'retrieval_method': args.retriever,
        'retrieval_topk': args.retrieval_topk,
        'test_sample_num': 1000,
    }

    config = Config(
        config_file_path='./flashrag/config/my_config.yaml',
        config_dict=config_dict
    )

    all_split = get_dataset(config)
    data = all_split[args.mode]

    prompt_templete = PromptTemplate(
        config,
        system_prompt="Answer the question based on the given document. \
                        Only give me the answer and do not output any other words. \
                        \nThe following are given documents.\n\n{reference}",
        user_prompt="Question: {question}\nAnswer:",
    )

    pipeline = SequentialPipeline(config, prompt_template=prompt_templete)
    output_dataset = pipeline.run(data, do_eval=True)
