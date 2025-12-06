import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import Union, List, Optional, Dict, Any
from datasets import Dataset, Value
from scipy.stats import spearmanr
from tqdm import tqdm
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    AutoModel, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorWithPadding
)
from flashrag.config import Config


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GlobalEncoder(nn.Module):
    def __init__(self, num_layer, global_hidden_size, num_heads, num_labels, dropout=0.0) -> None:
        super().__init__()
        self.trans_layer = nn.TransformerEncoderLayer(
            global_hidden_size,
            num_heads,
            batch_first=True,
            activation=F.gelu,
            norm_first=False,
            dropout=dropout,
        )
        self.context_attn = nn.TransformerEncoder(self.trans_layer, num_layer)
        self.query_emb = nn.Embedding(2, global_hidden_size)
        self.query_norm = nn.LayerNorm(global_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.score_linear = nn.Sequential(
            nn.Linear(global_hidden_size * 2, global_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_size, num_labels)
        )

    def forward(self, pair_feat, pair_nums, device):
        if len(set(pair_nums)) == 1:
            g_size = pair_nums[0]
            n_groups = len(pair_nums)
            # (N_groups, G_size, D)
            batch_pair_feat = pair_feat.view(n_groups, g_size, -1)
        else:
            # Fallback for variable group sizes
            batch_pair_feat = nn.utils.rnn.pad_sequence(
                pair_feat.split(pair_nums), batch_first=True
            )
        
        bs, max_len = batch_pair_feat.shape[:2]
        query_tags = torch.zeros(bs, max_len, dtype=torch.long, device=device)
        query_tags[:, 0] = 1
        batch_pair_feat = self.query_norm(batch_pair_feat + self.query_emb(query_tags))

        pair_list_feat = self.context_attn(
            batch_pair_feat,
            src_key_padding_mask=self.attention_mask(pair_nums).to(device),
            mask=self.query_isolated_attention_mask(pair_nums).to(device)
        )
        query_feat = pair_list_feat[:, 0, :]  # (B, D)
        
        if len(set(pair_nums)) == 1:
            g_size = pair_nums[0]
            # (B, G-1, D)
            ctx_feat = pair_list_feat[:, 1:g_size, :]
            # (B, G-1, D)
            query_expanded = query_feat.unsqueeze(1).expand(-1, g_size - 1, -1)
            # (B, G-1, 2D)
            pair_feat_cat = torch.cat([query_expanded, ctx_feat], dim=-1)
            # (B*(G-1), 2D)
            pair_feat_cat = pair_feat_cat.reshape(-1, pair_feat_cat.size(-1))
        else:
            pair_feat_list = [pair_list_feat[i, 1:n, :] for i, n in enumerate(pair_nums)]
            pair_feat_cat = torch.cat([
                torch.cat([query_feat[i].unsqueeze(0).repeat(n-1, 1), pair_feat_list[i]], dim=1)
                for i, n in enumerate(pair_nums)
            ], dim=0)

        pair_feat_cat = self.dropout(pair_feat_cat)
        logits = self.score_linear(pair_feat_cat)
        return logits, torch.cat([pair_list_feat[i, :n, :] for i, n in enumerate(pair_nums)], dim=0)

    def attention_mask(self, pair_num):
        max_len = max(pair_num)
        lengths = torch.tensor(pair_num)
        mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
        return mask

    def query_isolated_attention_mask(self, pair_num):
        max_len = max(pair_num)
        mask = torch.zeros(max_len, max_len, dtype=torch.bool)
        mask[0, 1:] = True
        return mask


class GroupBatchSampler(BatchSampler):
    def __init__(
        self, 
        dataset_len: int, 
        group_size: int = 11, 
        groups_per_batch: int = 1,
        shuffle: bool = True, 
        generator=None
    ):
        assert dataset_len % group_size == 0
        self.len = dataset_len
        self.g_size = group_size
        self.n_groups = dataset_len // group_size
        self.g_per_batch = groups_per_batch
        self.shuffle = shuffle
        self.gen = generator

    def __iter__(self):
        group_idx = list(range(self.n_groups))
        if self.shuffle:
            g = self.gen or torch.Generator()
            group_idx = [group_idx[i] for i in torch.randperm(self.n_groups, generator=g)]
        
        for i in range(0, self.n_groups, self.g_per_batch):
            batch_groups = group_idx[i:i+self.g_per_batch]
            yield [g * self.g_size + idx for g in batch_groups for idx in range(self.g_size)]

    def __len__(self):
        return math.ceil(self.n_groups / self.g_per_batch)


class CSM(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super().__init__()
        local_hidden_size = config['refiner_local_hidden_size']
        global_hidden_size = config['refiner_global_hidden_size']
        num_heads = config['refiner_num_heads']
        global_layers = config['refiner_global_layers']
        bert_path = config['model2path']['bert']

        self.num_labels = 1
        self.context_size = config['retrieval_topk']
        self.group_size = self.context_size + 1
        self.bert_path = bert_path
        
        self.local_encoder = AutoModel.from_pretrained(self.bert_path)
        self.feat_map = nn.Linear(local_hidden_size, global_hidden_size)
        self.global_encoder = GlobalEncoder(global_layers, global_hidden_size, num_heads, self.num_labels)
        self.proj_head = nn.Sequential(
            nn.Linear(global_hidden_size, global_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(global_hidden_size // 2, 64)
        )

        self.init_weights()
    
    def init_weights(self):
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        
        for block in [self.feat_map, self.global_encoder, self.proj_head]:
            block.apply(_init)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        device = input_ids.device
        self.global_encoder.device = device
        
        pooled = kwargs.pop("pooled", None)
        if pooled is None:
            out = self.local_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
            pooled = self.avg_pooling(out.last_hidden_state, attention_mask)
        else:
            out = None
        
        emb = self.feat_map(pooled)
        bs = len(input_ids)
        n_groups = bs // self.group_size
                
        emb = emb[:n_groups * self.group_size]
        logits, hidden_repr = self.global_encoder(emb, [self.group_size]*n_groups, device)
        
        cts_repr = torch.cat([hidden_repr[i*self.group_size+1 : (i+1)*self.group_size] for i in range(n_groups)], dim=0)
        cts_repr = F.normalize(self.proj_head(cts_repr), p=2, dim=1)
        cts_repr = cts_repr.view(n_groups, self.context_size, -1)

        if self.num_labels == 1:
            logits = torch.tanh(logits)

        return SequenceClassifierOutput(loss=None, logits=logits), cts_repr

    def avg_pooling(self, hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).to(hidden_state.dtype)
        sum_emb = torch.sum(hidden_state * mask, dim=1)
        sum_mask = mask.sum(dim=1)
        return sum_emb / sum_mask

    @staticmethod
    def get_group_sampler(
        dataset_len: int, 
        batch_size: int, 
        group_size: int = 11, 
        shuffle: bool = True
    ) -> GroupBatchSampler:
        g_per_batch = max(1, batch_size // group_size)
        return GroupBatchSampler(
            dataset_len=dataset_len,
            group_size=group_size,
            groups_per_batch=g_per_batch,
            shuffle=shuffle
        )

    def load_model(self, model_dir):
        ckpt_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Model checkpoint {ckpt_path} does not exist")

        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
        self.load_state_dict(state_dict)

    def fit_st(
        self,
        num_epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-4,
        cts_loss_w: float = 1,
        temp: float = 0.1,
        save_dir: Optional[str] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path, use_fast=True)
        group_size = self.group_size
        context_size = self.context_size
        
        for p in self.local_encoder.parameters():
            p.requires_grad = False
        
        train_ds = Dataset.from_parquet(f"{save_dir}/train_ds.parquet")

        train_args = TrainingArguments(
            output_dir=save_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size * group_size,
            per_device_eval_batch_size=batch_size * group_size,
            num_train_epochs=num_epochs,
            logging_steps=200,
            save_strategy="epoch",
            save_total_limit=1,
            dataloader_drop_last=True,
            lr_scheduler_type="cosine", 
            warmup_ratio=0.01,
            weight_decay=0.005,
            max_grad_norm=1.5,
            report_to=[],
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )

        class RegTrainer(Trainer):
            def get_train_dataloader(self):
                # sampler = CSM.get_group_sampler(len(self.train_dataset), self.args.train_batch_size)
                sampler = GroupBatchSampler()
                return DataLoader(
                    self.train_dataset, 
                    batch_sampler=sampler, 
                    collate_fn=self.data_collator,
                    num_workers=4,
                    pin_memory=True
                )

            def get_eval_dataloader(self, eval_dataset=None):
                eval_dataset = eval_dataset or self.eval_dataset
                sampler = CSM.get_group_sampler(len(eval_dataset), self.args.eval_batch_size, shuffle=False)
                return DataLoader(
                    eval_dataset, 
                    batch_sampler=sampler, 
                    collate_fn=self.data_collator,
                    num_workers=4,
                    pin_memory=True
                )

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs, cts_repr = model(**inputs)
                logits = outputs.logits.view(-1)
                target = labels.view(-1, group_size)[:, 1:].reshape(-1).to(logits.dtype)[:len(logits)]
                reg_loss = F.smooth_l1_loss(logits, target, beta=0.1)
                n_groups = len(logits) // context_size
                cts_loss = self.compute_cts_loss(cts_repr, target[:n_groups*context_size].view(n_groups, context_size)) if n_groups > 0 else 0

                loss = reg_loss + cts_loss_w * cts_loss
                return (loss, outputs) if return_outputs else loss

            def compute_cts_loss(self, cts_repr, targets):
                device = cts_repr.device
                signs = torch.sign(targets) # (B, N)
                pos_mask = signs > 1e-4 # (B, N)
                neg_mask = signs < -1e-4 # (B, N)
                valid_groups = pos_mask.any(dim=1) & neg_mask.any(dim=1)
                if not valid_groups.any():
                    return torch.tensor(0.0, device=device)
                
                cts_repr = cts_repr[valid_groups]   # (V, N, D)
                pos_mask = pos_mask[valid_groups]   # (V, N)
                neg_mask = neg_mask[valid_groups]   # (V, N)
                
                first_pos_idx = torch.argmax(pos_mask.long(), dim=1) # (V,)
                anchor = cts_repr[torch.arange(cts_repr.size(0)), first_pos_idx] # (V, D)
                
                # Similarity computation: (V, 1, D) @ (V, D, N) -> (V, 1, N) -> (V, N)
                sims = torch.bmm(anchor.unsqueeze(1), cts_repr.transpose(1, 2)).squeeze(1) / temp
                valid_mask = pos_mask | neg_mask
                sims_all = sims.clone()
                sims_all[~valid_mask] = float('-inf')
                log_sum_all = torch.logsumexp(sims_all, dim=1) # (V,)
                
                sims_pos = sims.clone()
                sims_pos[~pos_mask] = float('-inf')
                log_sum_pos = torch.logsumexp(sims_pos, dim=1) # (V,)
                
                loss = -log_sum_pos + log_sum_all
                return loss.mean()

        trainer = RegTrainer(
            model=self,
            args=train_args,
            train_dataset=train_ds,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True, max_length=512)
        )

        trainer.train()


    def fit_e2e(
        self,
        data_path: str,
        generator: Any,
        prompt_template: Any,
        num_epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        save_dir: Optional[str] = None,
        lambda_nec: float = 0.1,
        gumbel_temp: float = 1.0,
        grad_accum_steps: int = 1
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path, use_fast=True)
        context_size = self.context_size
        
        for p in self.local_encoder.parameters():
            p.requires_grad = False
        for p in generator.model.parameters():
            p.requires_grad = False
        generator.model.gradient_checkpointing_enable()
        generator.model.eval()

        class E2EDataset(torch.utils.data.Dataset):
            def __init__(self, data_path):
                with open(data_path, "r") as f:
                    data = json.load(f)
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                return {
                    "question": item["question"],
                    "answer": item.get("golden_answer")[0] if isinstance(item.get("golden_answer"), list) else item.get("golden_answer"),
                    "contexts": item["contexts"][:context_size] + [""]*(context_size - len(item["contexts"][:context_size]))
                }

        train_ds = E2EDataset(data_path)

        def gumbel_sample(logits, temp=1.0, hard=False):
            prob = (logits + 1) / 2
            eps = 1e-10
            logit = torch.log(prob + eps) - torch.log(1 - prob + eps)
            g1 = -torch.log(-torch.log(torch.rand_like(logit) + eps) + eps)
            g2 = -torch.log(-torch.log(torch.rand_like(logit) + eps) + eps)
            y = torch.sigmoid((logit + g1 - g2) / temp)
            return ((y > 0.5).float() - y.detach() + y) if hard else y

        def compute_e2e_loss(model, batch_data):
            gen_device = next(generator.model.parameters()).device
            csm_device = next(model.parameters()).device
            gen_embeds = generator.model.get_input_embeddings()
            
            qs, ans, ctxs_list = zip(*[(i["question"], i["answer"], i["contexts"]) for i in batch_data])
            
            # CSM Forward
            all_inputs = [t for q, ctxs in zip(qs, ctxs_list) for t in [q] + ctxs]
            csm_in = self.tokenizer(all_inputs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(csm_device)
            ctx_scores = model(**csm_in)[0].logits.squeeze(-1).view(len(batch_data), context_size)
            mask = gumbel_sample(ctx_scores, temp=gumbel_temp, hard=False).to(gen_device)
            mask_comp = 1 - mask

            suf_losses, nec_losses = [], []
            for i, (q, a, ctxs) in enumerate(zip(qs, ans, ctxs_list)):
                with torch.no_grad():
                    q_emb = gen_embeds(generator.tokenizer(prompt_template.get_string(question=q), return_tensors="pt", add_special_tokens=False).to(gen_device).input_ids)
                    ans_ids = generator.tokenizer.encode('Answer: ' + a, add_special_tokens=False, return_tensors="pt").to(gen_device)
                    ans_emb = gen_embeds(ans_ids)
                    c_embs = [gen_embeds(generator.tokenizer(f"\n{c}", return_tensors="pt", add_special_tokens=False).to(gen_device).input_ids) for c in ctxs]

                input_suf = torch.cat([q_emb] + [c * mask[i, j] for j, c in enumerate(c_embs)] + [ans_emb], dim=1)
                if input_suf.size(1) <= 2048:
                    labels = torch.full((1, input_suf.size(1)), -100, dtype=torch.long, device=gen_device)
                    labels[0, input_suf.size(1)-ans_ids.size(1):] = ans_ids[0]
                    suf_losses.append(generator.model(inputs_embeds=input_suf.detach(), labels=labels).loss)

                input_nec = torch.cat([q_emb] + [c * mask_comp[i, j] for j, c in enumerate(c_embs)], dim=1)
                logits = generator.model(inputs_embeds=input_nec.detach()).logits[0, -1, :]
                probs = F.log_softmax(logits, dim=-1)
                nec_losses.append(torch.sum(probs.exp() * probs))

            loss = (torch.stack(suf_losses).mean() if suf_losses else 0.0) + lambda_nec * (torch.stack(nec_losses).mean() if nec_losses else 0.0)
            return loss if isinstance(loss, float) else loss.to(gen_device)

        class E2ETrainer(Trainer):
            def __init__(self, use_mezo=True, **kwargs):
                super().__init__(**kwargs)
                self.use_mezo = use_mezo
            
            def get_train_dataloader(self):
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    sampler=RandomSampler(self.train_dataset),
                    collate_fn=lambda x: {"batch_data": x}
                )

            def get_eval_dataloader(self, eval_dataset=None):
                eval_dataset = eval_dataset or self.eval_dataset
                return DataLoader(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    sampler=SequentialSampler(eval_dataset),
                    collate_fn=lambda x: {"batch_data": x}
                )

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                loss = compute_e2e_loss(model, inputs["batch_data"])
                return (loss, None) if return_outputs else loss
            

        train_args = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=lr,
            weight_decay=0.005,
            logging_steps=100,
            save_strategy="epoch",
            report_to=[],
            fp16=True,
            warmup_steps=100,
            lr_scheduler_type="cosine",
        )

        trainer = E2ETrainer(
            use_mezo=True,  # Enable MeZO optimization
            model=self,
            args=train_args,
            train_dataset=train_ds,
            data_collator=lambda x: {"batch_data": x},
        )

        trainer.train()
        
        return trainer

    @torch.no_grad()
    def predict(
        self, 
        questions: List[str], 
        context_lists: List[List[str]], 
        batch_size: int = 32,
        device: torch.device = torch.device('cuda')
    ) -> List[List[float]]:
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path, use_fast=True)
            
        self.eval()
        self.to(device)
        
        all_scores = []
        for i in tqdm(range(0, len(questions), batch_size), desc="CSM Predicting"):
            batch_questions = questions[i : i + batch_size]
            batch_context_lists = context_lists[i : i + batch_size]
            
            flat_texts = [t for q, ctxs in zip(batch_questions, batch_context_lists) for t in [q] + ctxs]
            pair_nums = [len(ctxs) + 1 for ctxs in batch_context_lists]
            
            encode_bs, pooled_list = 256, []
            for j in range(0, len(flat_texts), encode_bs):
                inputs = self.tokenizer(flat_texts[j : j + encode_bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                out = self.local_encoder(**inputs, return_dict=True)
                pooled_list.append(self.avg_pooling(out.last_hidden_state, inputs["attention_mask"]))
            
            logits, _ = self.global_encoder(self.feat_map(torch.cat(pooled_list, dim=0)), pair_nums, device)
            logits = (torch.tanh(logits) if self.num_labels == 1 else logits).squeeze(-1).cpu().tolist()
            logits = [logits] if isinstance(logits, float) else logits
            
            current_idx = 0
            for n in pair_nums:
                all_scores.append(logits[current_idx : current_idx + n - 1])
                current_idx += n - 1
                
        return all_scores