from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.config import Config
from flashrag.prompt.utils import get_generation_config
import argparse
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--root_dir', type=str)
    args.add_argument('--data_name', type=str, default='nq',
                      choices=['nq', 'triviaqa', 'webqa', 'hotpotqa', '2wikimultihopqa', 'asqa', 'fever', 'truthful_qa'])
    args.add_argument('--generator', type=str, default='llama3-8B-instruct')
    args.add_argument('--gpu_id', type=str, default='2')
    args.add_argument('--mode', type=str, default='test')
    args.add_argument('--retriever', type=str, default='e5')
    args.add_argument('--refiner', type=str, default='csm')
    args.add_argument('--retrieval_topk', type=int, default=10)
    args = args.parse_args()
    
    data_name = args.data_name
    root_dir = args.root_dir
    gen_dir = 'llama'
    model2path = {
        'e5': f'{root_dir}/ckpt/e5-base-v2',
        'llama3-8B-instruct': f'{root_dir}/ckpt/Llama-3.1-8B-Instruct',
        'bert': f'{root_dir}/ckpt/bert-base-uncased',
        'csm': f'{root_dir}/ckpt/csm/{data_name}_{gen_dir}',
    }

    system_prompt, user_prompt, max_new_tokens, metrics, gen_batch_size = \
          get_generation_config(data_name)

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
        'save_retrieval_cache': False,
        'use_retrieval_cache': True,
        'retrieval_cache_path': f'{root_dir}/data/csm/retrieval_cache/{data_name}_{args.mode}.json',
        'retrieval_method': args.retriever,
        'retrieval_topk': args.retrieval_topk,
        # --- refiner ---
        'refiner_name': args.refiner,
        'refiner_score_path': f'{root_dir}/data/csm/refiner_scores/{gen_dir}/{data_name}_{args.mode}_ci.json',
        # --- csm model conf ---
        'refiner_local_hidden_size': 768,
        'refiner_global_layers': 3,
        'refiner_global_hidden_size': 1024,
        'refiner_num_heads': 32,
        # --- generator ---
        'generator_model': args.generator,
        'generator_batch_size': gen_batch_size,
        'generator_max_input_len': 2048,
        'metrics': metrics,
        'max_new_tokens': max_new_tokens,
        'save_intermediate_data': False,
        'test_sample_num': 1000,
    }
    print('---runing context selection rag---')
    print(f'---dataset: {data_name}, generator: {args.generator}---')

    config = Config(
        config_file_path='./flashrag/config/my_config.yaml',
        config_dict=config_dict
    )

    all_split = get_dataset(config)
    data = all_split[args.mode]

    prompt_templete = PromptTemplate(
        config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    pipeline = SequentialPipeline(config, prompt_template=prompt_templete)
    output_dataset = pipeline.run(data, do_eval=True)
    print('-----------')
