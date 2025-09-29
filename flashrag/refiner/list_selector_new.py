from datasets import Dataset, Features, Value, Sequence
import numpy as np
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
import torch.nn as nn
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from copy import deepcopy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CrossEncoder(nn.Module):
    def __init__(self, bert_path, num_labels, num_global_layers, lora_config, drop_rate, loss_fct=None):
        super().__init__()
        self.num_labels = num_labels
        self.drop_rate = drop_rate

        self.bert = AutoModel.from_pretrained(bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.hidden_size = self.bert.config.hidden_size

        if lora_config:
            self.bert = get_peft_model(self.bert, lora_config)
            # self.bert.print_trainable_parameters()

        self.global_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=4,
                dim_feedforward=self.hidden_size,
                batch_first=True,
                dropout=drop_rate
            ),
            num_layers=num_global_layers
        )
        self.head = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        self.projector = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_size, 32)
        )
        self.loss_fct = loss_fct

    def forward(self, input_ids, attention_mask, labels=None, *args, **kwargs):
        # input shape: [batch, num_contexts, seq_len]
        batch_size = input_ids.size(0)
        num_contexts = input_ids.size(1)
        embs = self.bert(
            input_ids=input_ids.view(batch_size * num_contexts, -1),
            attention_mask=attention_mask.view(batch_size * num_contexts, -1)
        ).last_hidden_state[:, 0]  # [batch*num_contexts, hidden]
        embs = embs.view(batch_size, num_contexts, -1)  # [batch, num_contexts, hidden]
        embs = self.global_encoder(embs)  # [batch, num_contexts, hidden]
        cts_embs = self.projector(embs)  # [batch, num_contexts, 32]
        outputs = self.head(embs)
        if labels is not None:
            if kwargs:
                gen_input_ids = kwargs['gen_input_ids']
                gen_context_indicies = kwargs['gen_context_indicies']
                gen_attention_mask = kwargs['gen_attention_mask']
                loss = self.loss_fct(outputs, labels, gen_input_ids, gen_attention_mask, gen_context_indicies)
            else:
                if self.num_labels == 1:
                    outputs = outputs.tanh().view(-1)
                    loss = self.loss_fct(cts_embs, outputs, labels.view(-1))
                else:
                    outputs = outputs.softmax(dim=1).view(-1, self.num_labels)
                    loss = self.loss_fct(cts_embs, outputs, labels.view(-1).long())
            return {"loss": loss, "logits": outputs}
        return {"logits": outputs}

    def load_model(self, save_path):
        model_files = ['pytorch_model.bin', 'model.safetensors']
        for model_file in model_files:
            model_path = os.path.join(save_path, model_file)
            if os.path.exists(model_path):
                try:
                    if model_file.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        model_state_dict = load_file(model_path)
                    else:
                        model_state_dict = torch.load(model_path)
                    self.load_state_dict(model_state_dict)
                    return self
                except Exception as e:
                    print(f"Failed to load {model_file}: {str(e)}")
                    continue
        raise FileNotFoundError(f"No valid model file found in {save_path}")


def contrastive_loss(embs, labels, temperature=0.1, eps=1e-12):
    """
    Args:
        embs: Tensor [batch_size, num_ctx, hidden_dim] (e.g. 32)
        labels: Tensor [batch_size, num_ctx, 1]
    """
    batch_size, num_ctx, hidden_dim = embs.shape
    labels = (labels > 0).float()
    labels[labels == 0] = -1
    # Flatten to [batch_size*num_ctx, hidden_dim]
    flat_embs = embs.view(-1, hidden_dim)
    flat_labels = labels.view(-1, 1)
    total_samples = flat_embs.size(0)

    # Calculate cosine similarity matrix [total_samples, total_samples]
    sim_matrix = F.cosine_similarity(
        flat_embs.unsqueeze(1),
        flat_embs.unsqueeze(0),
        dim=-1
    ) / temperature
    sign_matrix = torch.sign(flat_labels @ flat_labels.T)
    question_mask = torch.block_diag(
        *[torch.ones(num_ctx, num_ctx) for _ in range(batch_size)]
    ).bool().to(embs.device)

    pos_mask = (sign_matrix > 0) & (~torch.eye(total_samples, dtype=torch.bool, device=embs.device))
    neg_mask = (sign_matrix < 0)
    intra_pos_mask = pos_mask & question_mask     # Same question positive
    inter_pos_mask = pos_mask & ~question_mask    # Cross question positive
    intra_neg_mask = neg_mask & question_mask     # Same question negative
    inter_neg_mask = neg_mask & ~question_mask    # Cross question negative
    weighted_sim_exp = sim_matrix.exp()
    weighted_pos = (intra_pos_mask.float() + 2.0 * inter_pos_mask.float()) * weighted_sim_exp
    weighted_neg = (2.0 * intra_neg_mask.float() + inter_neg_mask.float()) * weighted_sim_exp

    numerator = weighted_pos.sum(dim=1) + eps
    denominator = weighted_neg.sum(dim=1) + numerator + eps
    valid = (weighted_pos.sum(dim=1) > 0)
    losses = torch.zeros_like(numerator)
    losses[valid] = -torch.log(numerator[valid] / denominator[valid])

    if valid.sum() == 0:
        return torch.tensor(0.0, device=embs.device)
    return losses[valid].mean()


class SftCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, embs, y_pred, y_true):
        loss = self.loss_fct(y_pred, y_true)
        cts_loss = contrastive_loss(embs, y_true.unsqueeze(-1))
        return loss + 0.1 * cts_loss


class SftMSELoss(nn.Module):
    def __init__(self, y_all, num_bins=100):
        super().__init__()
        self.num_bins = num_bins
        self.bins, self.weights = self.compute_global_weights(y_all)

    def compute_global_weights(self, y_all):
        # y_all = torch.tensor(y_all)
        bins = torch.linspace(-1, 1, self.num_bins + 1)
        hist = torch.histc(y_all.view(-1), bins=self.num_bins, min=-1, max=1)
        weights = 1.0 / (hist + 1e-8)
        weights = weights / weights.sum() * 10
        return bins, weights

    def forward(self, embs, y_pred, y_true):
        indices = torch.bucketize(y_true.detach().cpu(), self.bins, right=True)
        indices = torch.clamp(indices - 1, 0, self.num_bins - 1)
        weights = self.weights[indices]
        squared_error = (y_pred - y_true) ** 2
        weighted_loss = squared_error * weights.to(y_pred.device)
        sum_weights = weights.sum() + 1e-8
        mse_loss = weighted_loss.mean() / sum_weights
        sign_loss = (-y_true * y_pred).relu().mean()
        cts_loss = contrastive_loss(embs, y_true.unsqueeze(-1))
        return mse_loss + 0.1 * cts_loss + sign_loss


class LOOLoss(nn.Module):
    def __init__(self, generator, prompt_template):
        super().__init__()
        self.generator = generator
        self.prompt_template = prompt_template

    def info_loss_reg(self, att, alpha=0.1, beta=0.1):
        EPS = 1e-6
        info_loss = alpha * att.mean()
        ent = -att * torch.log(att + EPS) - (1 - att) * torch.log(1 - att + EPS)
        info_loss = info_loss + beta * ent.mean()
        return info_loss

    @staticmethod
    def sampling(att, training):
        temp = 1
        if training:
            random_noise = torch.empty_like(att).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att + random_noise) / temp).sigmoid()
        else:
            att_bern = (att).sigmoid()
        return att_bern

    def forward(self, context_scores, labels, input_ids, attention_mask, context_indicies):
        num_samples = len(context_indicies)
        num_contexts = len(context_indicies[0])-1
        emb_layer = self.generator.model.get_input_embeddings()
        embs = emb_layer(input_ids)
        emb_scores = torch.ones_like(embs)
        context_scores = self.sampling(context_scores.view(num_samples, -1), training=True)
        context_scores = context_scores.view(num_samples, num_contexts, -1)
        for i in range(num_samples):
            for k in range(num_contexts):
                emb_scores[i, context_indicies[i][k]:context_indicies[i][k+1], :] = context_scores[i, k]
        pred_loss = self.generator.model(
            inputs_embeds=embs*emb_scores, attention_mask=attention_mask, labels=labels
        ).loss
        # pred_loss = self.generator.model(
        #     input_ids=input_ids, attention_mask=attention_mask, labels=labels, input_scores=emb_scores
        # ).loss
        info_loss = []
        context_scores = context_scores.view(num_samples, -1)
        for i in range(num_samples):
            att = context_scores[i]
            info_loss.append(self.info_loss_reg(att))
        info_loss = torch.tensor(info_loss).mean().to(input_ids.device)
        loss = pred_loss + info_loss
        return loss


class ListSelector():
    def __init__(self, config, num_global_layers=1, drop_rate=0.5, generator=None, prompt_template=None):
        self.config = config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules=["query", "value"],
            modules_to_save=["global_encoder", "head", "projector"]
        )
        bert_path = config['model2path']['bert']
        self.save_path = config['model2path']['list_selector']
        self.selector = config['refiner_selector']
        self.num_labels = 2 if 'cls' in self.selector else 1
        self.model = CrossEncoder(bert_path, self.num_labels, num_global_layers, lora_config, drop_rate)
        self.generator = generator
        self.prompt_template = prompt_template

    def model_init(self):
        return self.model

    def _down_sampling(self, questions, context_lists, labels):
        abs_max = torch.abs(torch.tensor(labels)).max(dim=1).values
        mask = abs_max <= 0.1
        near_zero_indices = torch.where(mask)[0]
        other_indices = torch.where(~mask)[0]
        target_ratio = 0.3
        num_to_keep = int(len(near_zero_indices) * target_ratio)
        if num_to_keep < len(near_zero_indices):
            keep_indices = torch.randperm(len(near_zero_indices))[:num_to_keep]
            selected_indices = near_zero_indices[keep_indices]
        else:
            selected_indices = near_zero_indices
        final_indices = torch.cat([selected_indices, other_indices])
        questions = [questions[i] for i in final_indices]
        context_lists = [context_lists[i] for i in final_indices]
        labels = [labels[i] for i in final_indices]
        return questions, context_lists, labels

    def _piecewise_scale(self, x, linear_range=1.0):
        x = torch.tensor(x)
        scaled = torch.zeros_like(x)
        mask_linear = torch.abs(x) <= linear_range
        scaled[mask_linear] = x[mask_linear] / linear_range
        mask_pos = x > linear_range
        scaled[mask_pos] = 1 - torch.log(x[mask_pos] - linear_range + 1) / 10
        mask_neg = x < -linear_range
        scaled[mask_neg] = -1 + torch.log(-x[mask_neg] - linear_range + 1) / 10
        return torch.clip(scaled, -1, 1).cpu().numpy().tolist()

    def _preprocess_function(self, examples, max_length):
        tokenizer = self.model.tokenizer
        # flatten to (question, context) pairs
        flat_questions = [q for q in examples["question"] for _ in range(10)]
        flat_contexts = [c for ctxs in examples["contexts"] for c in ctxs]
        num_samples = len(examples["question"])
        num_contexts = len(examples["contexts"][0])
        # Tokenize
        tokenized = tokenizer(
            flat_questions,
            flat_contexts,
            truncation=True,
            padding='max_length',
            return_tensors="pt",
            max_length=max_length
        )
        if 'sft' in self.selector:
            if "labels" in examples:
                labels = torch.tensor(examples["labels"])
                if "cls" in self.selector:
                    labels = torch.where(labels > 0, torch.ones_like(labels), torch.zeros_like(labels))
                return {
                    "input_ids": tokenized["input_ids"].view(num_samples, num_contexts, -1),
                    "attention_mask": tokenized["attention_mask"].view(num_samples, num_contexts, -1),
                    "labels": labels
                }
            else:
                return {
                    "input_ids": tokenized["input_ids"].view(num_samples, num_contexts, -1),
                    "attention_mask": tokenized["attention_mask"].view(num_samples, num_contexts, -1)
                }
        else:
            gen_tokenizer = self.generator.tokenizer
            golden_answers = examples['labels']
            prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=c)
                for q, c in zip(examples['question'], examples['contexts'])
            ]
            # TODO consider the case where there are multiple answers
            flat_golden_answers = [ans[0] for ans in golden_answers]
            input_texts = [p + ans for p, ans in zip(prompts, flat_golden_answers)]
            gen_tokens = gen_tokenizer(
                input_texts, return_tensors='pt',
                padding=True, return_offsets_mapping=True
            )
            out_mask = gen_tokenizer(flat_golden_answers, return_tensors='pt', padding=True).attention_mask
            context_indicies = [
                self.generator.relocate_chunk_index(text, mapping)
                for text, mapping in zip(input_texts, gen_tokens.offset_mapping)
            ]
            target_ids = deepcopy(gen_tokens.input_ids)
            for i, mask_len in enumerate(out_mask.sum(dim=1)):
                target_ids[i, :-mask_len] = -100
            return {
                "input_ids": tokenized["input_ids"].view(num_samples, num_contexts, -1),
                "attention_mask": tokenized["attention_mask"].view(num_samples, num_contexts, -1),
                "gen_input_ids": gen_tokens["input_ids"],
                "gen_attention_mask": gen_tokens["attention_mask"],
                "labels": target_ids,
                "gen_context_indicies": torch.tensor(context_indicies),
            }

    def get_dataset(self, questions, context_lists, labels=None):
        num_contexts = len(context_lists[0])
        data = {'question': questions, 'contexts': context_lists}
        features = {
            'question': Value('string'),
            'contexts': Sequence(Value('string'), length=num_contexts)
        }
        if labels is not None:
            data['labels'] = labels
            if self.selector == 'e2e':
                features['labels'] = Sequence(Value('string'))
            else:
                features['labels'] = Sequence(Value('float32'), length=num_contexts)
        dataset = Dataset.from_dict(data, features=Features(features))
        return dataset.map(
            lambda x: self._preprocess_function(x, max_length=512),
            batched=True,
            remove_columns=dataset.column_names
        )

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if "reg" in self.selector or "e2e" in self.selector:
            mse = np.mean((predictions.reshape(-1) - labels.reshape(-1)) ** 2)
            sign_acc = np.mean(np.sign(predictions.reshape(-1)) == np.sign(labels.reshape(-1)))
            return {"eval_mse": mse, "eval_sign_acc": sign_acc}
        else:
            predictions = np.argmax(predictions, axis=-1)
            accuracy = np.mean(predictions == labels)
            return {"eval_acc": accuracy}

    def get_loss_fct(self, labels):
        if 'sft' in self.selector:
            labels = torch.tensor(labels)
            if self.num_labels == 1:
                loss_fct = SftMSELoss(labels)
            else:
                loss_fct = SftCELoss()
        else:
            loss_fct = LOOLoss(self.generator, self.prompt_template)
        return loss_fct

    def fit(self, train_data, num_epochs=20, batch_size=2, lr=1e-5):
        questions, context_lists, labels = train_data
        if 'sft' in self.selector:
            labels = self._piecewise_scale(labels)
            questions, context_lists, labels = self._down_sampling(questions, context_lists, labels)
        dataset = self.get_dataset(questions, context_lists, labels)
        dataset = dataset.train_test_split(test_size=0.1)
        self.model.loss_fct = self.get_loss_fct(labels)

        training_args = TrainingArguments(
            output_dir=self.save_path,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            num_train_epochs=num_epochs,
            bf16=True,
            weight_decay=0.01,
            # evaluation_strategy="epoch",
            # # save_strategy="epoch",
            # logging_strategy="epoch",
            # eval_strategy="steps",
            # eval_steps=100,
            logging_strategy="steps",
            logging_steps=100,
            # load_best_model_at_end=True,
            save_total_limit=1,
            remove_unused_columns=False,
        )
        if "reg" in self.selector:
            training_args.metric_for_best_model = "eval_mse"
        else:
            training_args.metric_for_best_model = "eval_acc"
            training_args.greater_is_better = True

        trainer = Trainer(
            model=self.model,
            # model_init=self.model_init,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=self.compute_metrics
        )
        trainer.train()
        # best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
        # print(best_run)
        # for n, v in best_run.hyperparameters.items():
        #     setattr(trainer.args, n, v)
        # trainer.train()
        # outputs = trainer.predict(dataset["test"])
        # predictions = np.argmax(outputs[0], axis=-1)
        trainer.save_model(self.save_path)

    @torch.no_grad()
    def predict(self, test_data, labels=None, batch_size=64):
        self.model.load_model(self.save_path)
        self.model.eval()
        if not labels:
            test_dataset = self.get_dataset(test_data[0], test_data[1])
        else:
            test_dataset = self.get_dataset(test_data[0], test_data[1], labels)
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=self.save_path,
                per_device_eval_batch_size=batch_size,
                fp16=True
            ),
        )
        num_samples = len(test_data[0])
        if labels is not None:
            if 'sft' in self.selector:
                labels = self._piecewise_scale(labels)
            self.model.loss_fct = self.get_loss_fct(labels)
            outputs = trainer.predict(test_dataset)
            metrics = self.compute_metrics((outputs.predictions, outputs.label_ids))
            predictions = outputs.predictions.reshape(num_samples, -1).tolist()
            print(f"eval_mse: {metrics['eval_mse']}, eval_sign_acc: {metrics['eval_sign_acc']}")
            return predictions
        else:
            predictions = trainer.predict(test_dataset).predictions.reshape(num_samples, -1).tolist()
            return predictions
