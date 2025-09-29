import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from transformers import AdamW, get_scheduler


class SelectorDataset(Dataset):
    def __init__(self, questions, context_lists, labels, task_type='regression'):
        self.questions = questions
        self.context_lists = context_lists
        self.task_type = task_type
        self.labels = labels
    
    def label_scaling(self):
        # self.labels = linear_scale(self.labels)
        labels = piecewise_scale(torch.tensor(self.labels))
        self.labels = labels.numpy().tolist()
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):   
        return {
            'questions': self.questions[idx],
            'contexts': self.context_lists[idx],
            'labels': self.labels[idx]
        }


def linear_scale(x):
    max_abs = torch.max(torch.abs(x))
    if max_abs == 0:
        return x
    scaled_data = x / max_abs
    return scaled_data


def piecewise_scale(x, linear_range=1.0):
    scaled = torch.zeros_like(x)
    mask_linear = torch.abs(x) <= linear_range
    scaled[mask_linear] = x[mask_linear] / linear_range
    mask_pos = x > linear_range
    scaled[mask_pos] = 1 - torch.log(x[mask_pos] - linear_range + 1) / 10
    mask_neg = x < -linear_range
    scaled[mask_neg] = -1 + torch.log(-x[mask_neg] - linear_range + 1) / 10
    return torch.clip(scaled, -1, 1)


def custom_collate_fn(batch):
    batch_questions = [item['questions'] for item in batch]
    batch_contexts = [item['contexts'] for item in batch]
    if not isinstance(batch[0]['labels'][0], str):
        batch_labels = torch.tensor([item['labels'] for item in batch])
    else:
        batch_labels = [item['labels'] for item in batch]
    return {
        'questions': batch_questions,
        'contexts': batch_contexts,
        'labels': batch_labels
    }


class WeightedMSELoss((nn.Module)):
    def __init__(self, y_all, num_bins=100):
        super().__init__()
        self.num_bins = num_bins
        self.bins, self.weights = self._compute_global_weights(y_all)
    
    def _compute_global_weights(self, y_all):
        y_all = torch.tensor(y_all)
        bins = torch.linspace(-1, 1, self.num_bins + 1)
        hist = torch.histc(y_all.view(-1), bins=self.num_bins, min=-1, max=1)
        weights = 1.0 / (hist + 1e-8)
        weights = weights / weights.sum() * 10
        return bins, weights
    
    def forward(self, y_pred, y_true, alpha=0.01):
        indices = torch.bucketize(y_true.detach().cpu(), self.bins, right=True)
        indices = torch.clamp(indices - 1, 0, self.num_bins - 1)
        weights = self.weights[indices]
        squared_error = (y_pred - y_true) ** 2
        weighted_loss = squared_error * weights.to(y_pred.device)
        sum_weights = weights.sum() + 1e-8
        mse_loss = weighted_loss.mean() / sum_weights
        return mse_loss
        # # sign_loss = F.relu(-y_pred * y_true).mean()
        # sign_loss = (torch.sign(y_pred) != torch.sign(y_true)).float().mean()
        # return mse_loss + alpha * sign_loss


class SignLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        loss = torch.sign(y_pred) != torch.sign(y_true)
        if self.reduction == 'mean':
            return loss.float().mean()
        elif self.reduction == 'none':
            return loss.float()


class CrossEncoder(nn.Module):
    def __init__(self, config, task_type='regression'):
        super().__init__()
        self.config = config
        self.device = 'cuda'
        self.num_global_layers = 1
        self.task_type = task_type
        self.dropout = 0.3

        # Initialize BERT
        bert_path = config['model2path']['bert']
        self.bert = AutoModel.from_pretrained(bert_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.hidden_size = self.bert.config.hidden_size
        
        # Initialize global layers
        self.global_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=4,
            dim_feedforward=self.hidden_size * 2,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
            dropout=self.dropout,
        ).to(self.device)
        
        self.global_encoder = nn.TransformerEncoder(
            self.global_layer,
            num_layers=self.num_global_layers
        ).to(self.device)
        
        # regression head
        self.regression_head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        ).to(self.device)
        
        # classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 2),
        ).to(self.device)
    
    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @torch.no_grad()
    def encode_pairs(self, questions, contexts, batch_size=16):
        inputs = self.tokenizer(
            questions, contexts, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        outputs = self.bert(**inputs)
        # return outputs.last_hidden_state[:, 0, :]
        # average pooling
        return self.average_pool(outputs.last_hidden_state, inputs.attention_mask)

    def forward(self, questions, context_lists):
        batch_size = len(questions)
        num_docs = len(context_lists[0])
        flat_questions = [q for q in questions for q in [q]*num_docs]
        flat_contexts = [c for ctxs in context_lists for c in ctxs]
        embs = self.encode_pairs(flat_questions, flat_contexts)  # [batch_size*num_docs, hidden_size]
        embs = embs.view(batch_size, num_docs, self.hidden_size)    # [batch_size, num_docs, hidden_size]
        embs = self.global_encoder(embs)  # [batch_size, num_docs, hidden_size]
        if self.task_type == 'regression':
            scores = self.regression_head(embs).squeeze(-1)  # [batch_size, num_docs]
        else:
            scores = self.classification_head(embs)  # [batch_size, num_docs, 2]
        return scores
    

class ListSelector(nn.Module):
    def __init__(self, config, generator=None, prompt_template=None):
        super().__init__()
        self.device = 'cuda'
        self.num_bins = 100 # for weighted mse loss and down sampling
        if 'reg' in config['refiner_selector'] or 'e2e' in config['refiner_selector']:
            self.task_type = 'regression'
        else:
            self.task_type = 'classification'
        self.model = CrossEncoder(config, self.task_type)
        self.generator = generator
        self.prompt_template = prompt_template
    
    def _down_sampling(self, dataset):
        labels = dataset.labels  # [n, 10]
        abs_mean = torch.abs(labels).max(dim=1).values
        mask = abs_mean <= 0.1
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
        dataset.questions = [dataset.questions[i] for i in final_indices]
        dataset.context_lists = [dataset.context_lists[i] for i in final_indices]
        dataset.labels = dataset.labels[final_indices]
        return dataset
    
    def _sft_loss(self, outputs, labels, criterion):
        if self.task_type == 'regression':
            outputs = torch.tanh(outputs)
            loss = criterion(outputs.to(self.device), labels.to(self.device))
        else:
            # outputs = torch.softmax(outputs, dim=1)
            labels = torch.where(labels < 0, torch.zeros_like(labels), torch.ones_like(labels)).long()
            loss = criterion(outputs.permute(0,2,1).to(self.device), labels.to(self.device))
        return loss
    
    def fit_sft(self, train_dataset, dev_dataset, num_epochs, batch_size, lr):
        """Train the selector model with SFT"""
        self.train()
        if self.task_type == 'regression':
            train_dataset = self._down_sampling(train_dataset)
        scaler = GradScaler()
        optimizer = AdamW(self.parameters(), lr=lr)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, 
                                     num_training_steps=num_epochs*len(train_dataloader))
        
        best_dev_loss = float('inf')
        best_model = None
        
        # Training loop
        criterion = WeightedMSELoss(train_dataset.labels, self.num_bins) if self.task_type == 'regression' else nn.CrossEntropyLoss()
        sign_criterion = SignLoss()
        for epoch in range(num_epochs):
            start_time = time.time()
            total_train_sft_loss = 0
            total_train_sign_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                with autocast():
                    outputs = self.model.forward(batch['questions'], batch['contexts'])
                    sft_loss = self._sft_loss(outputs, batch['labels'], criterion)
                    total_train_sft_loss += sft_loss.detach().cpu().item()
                    loss = sft_loss
                    if self.task_type == 'regression':
                        sign_loss = sign_criterion(outputs.to(self.device), batch['labels'].to(self.device))
                        loss = sft_loss + 0.1 * sign_loss
                        total_train_sign_loss += sign_loss.detach().cpu().item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                torch.cuda.empty_cache()
            
            # validation
            total_dev_sft_loss = 0
            total_dev_sign_loss = 0
            criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
            sign_loss = SignLoss()
            with torch.no_grad():
                for batch in dev_dataloader:
                    outputs = self.model.forward(batch['questions'], batch['contexts'])
                    sft_loss = self._sft_loss(outputs, batch['labels'], criterion)
                    total_dev_sft_loss += sft_loss.detach().cpu().item()
                    if self.task_type == 'regression':
                        sign_loss = sign_criterion(outputs.to(self.device), batch['labels'].to(self.device))
                        # loss = loss + 0.1 * sign_loss
                        total_dev_sign_loss += sign_loss.detach().cpu().item()
                    
            avg_train_sft_loss = total_train_sft_loss / len(train_dataloader)
            avg_train_sign_loss = total_train_sign_loss / len(train_dataloader)
            avg_dev_sft_loss = total_dev_sft_loss / len(dev_dataloader)
            avg_dev_sign_loss = total_dev_sign_loss / len(dev_dataloader)
            print(f'Epoch {epoch+1}, Average Train SFT Loss: {avg_train_sft_loss:.4f}, Average Train Sign Loss: {avg_train_sign_loss:.4f}, '
                  f'Average Dev SFT Loss: {avg_dev_sft_loss:.4f}, Average Dev Sign Loss: {avg_dev_sign_loss:.4f}, '
                  f'Time: {time.time() - start_time:.2f}s')
            
            if avg_dev_sft_loss < best_dev_loss:
                best_dev_loss = avg_dev_sft_loss
                best_model = self.state_dict()
        
        self.load_state_dict(best_model)
        self.plot_loss_distribution(train_dataloader, nn.MSELoss(reduction='none'), 
                                    f'./visualization/{self.task_type}_loss_distribution_train.png')
        self.plot_loss_distribution(dev_dataloader, nn.MSELoss(reduction='none'), 
                                    f'./visualization/{self.task_type}_loss_distribution_valid.png')
        if self.task_type == 'regression':
            self.plot_loss_distribution(train_dataloader, SignLoss(reduction='none'), 
                                    f'./visualization/{self.task_type}_sign_loss_distribution_train.png')
            self.plot_loss_distribution(dev_dataloader, SignLoss(reduction='none'), 
                                    f'./visualization/{self.task_type}_sign_loss_distribution_valid.png')

    @torch.no_grad()
    def plot_loss_distribution(self, dataloader, criterion, save_path):
        self.eval()
        train_losses = torch.tensor([])
        for batch in dataloader:
            outputs = self.model.forward(batch['questions'], batch['contexts'])
            loss = self._sft_loss(outputs, batch['labels'], criterion).cpu()
            batch_losses = torch.cat([batch['labels'].view(1, -1), loss.view(1, -1)], dim=0).T.detach().cpu()
            train_losses = torch.cat([train_losses, batch_losses], dim=0)
        train_losses = train_losses.numpy()
        # plot loss distribution
        plt.figure(figsize=(10, 5))
        x = train_losses[:, 0]
        y = train_losses[:, 1]
        plt.scatter(x, y, label='Loss', alpha=0.5)
        plt.xlabel('Labels')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path)

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
    
    @staticmethod
    def info_loss_reg(att, alpha=0.1, beta=0.1):
        EPS = 1e-6
        info_loss = alpha * att.mean()
        ent = -att * torch.log(att + EPS) - (1 - att) * torch.log(1 - att + EPS)
        info_loss = info_loss + beta * ent.mean()
        return info_loss
    
    def fit_e2e(self, train_dataset, dev_dataset, generator, prompt_template, num_epochs, batch_size, lr):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scaler = GradScaler()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        best_dev_loss = float('inf')
        best_model = None

        for epoch in range(num_epochs):
            total_train_loss = 0
            for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                optimizer.zero_grad()
                with autocast():
                    questions = batch['questions']
                    context_lists = batch['contexts']
                    golden_answers = batch['labels']
                    context_scores = self.model.forward(questions, context_lists)
                    context_scores = self.sampling(context_scores, training=True)
                    batch_prompt = [
                        prompt_template.get_string(question=q, retrieval_result=r)
                        for q, r in zip(questions, context_lists)
                    ]
                    pred_loss = generator.cal_pred_loss_w_context(batch_prompt, golden_answers, context_scores)
                    info_loss = []
                    for i in range(batch_size):
                        info_loss.append(self.info_loss_reg(context_scores[i]))
                    info_loss = torch.stack(info_loss)
                    loss = pred_loss.mean() + info_loss.mean()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.empty_cache()
                total_train_loss += loss.detach().cpu().item()
            avg_train_loss = total_train_loss / len(train_dataloader)

            # validation
            total_dev_loss = 0
            with torch.no_grad(), autocast():
                for batch in dev_dataloader:
                    questions = batch['questions']
                    context_lists = batch['contexts']
                    golden_answers = batch['labels']
                    context_scores = self.model.forward(questions, context_lists)
                    context_scores = self.sampling(context_scores, training=False)
                    batch_prompt = [
                        prompt_template.get_string(question=q, retrieval_result=r)
                        for q, r in zip(questions, context_lists)
                    ]
                    pred_loss = generator.cal_pred_loss_w_context(batch_prompt, golden_answers, context_scores)
                    total_dev_loss += pred_loss.mean().detach().cpu().item()
            avg_dev_loss = total_dev_loss / len(dev_dataloader)
            if avg_dev_loss < best_dev_loss:
                best_dev_loss = avg_dev_loss
                best_model = self.state_dict()

            print(f'Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}, Average Dev Loss: {avg_dev_loss:.4f}')

        self.load_state_dict(best_model)

    def fit(self, train_data, num_epochs=100, batch_size=64, lr=2e-4, paradigm='e2e'):
        generator = self.generator
        prompt_template = self.prompt_template
        questions, context_lists, labels = train_data
        split_idx = int(len(questions) * 0.8)
        
        train_dataset = SelectorDataset(questions[:split_idx], context_lists[:split_idx], labels[:split_idx], self.task_type)
        dev_dataset = SelectorDataset(questions[split_idx:], context_lists[split_idx:], labels[split_idx:], self.task_type)
        if paradigm == 'e2e':
            self.fit_e2e(train_dataset, dev_dataset, generator, prompt_template, num_epochs, batch_size=4, lr=lr)
        else:
            if self.task_type == 'regression':
                train_dataset.label_scaling()
                dev_dataset.label_scaling()
            self.fit_sft(train_dataset, dev_dataset, num_epochs, batch_size, lr)
    
    @torch.inference_mode()
    def predict(self, questions, context_lists, batch_size=64):
        """Predict scores for question-context pairs"""
        self.eval()
        scores = []
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_contexts = context_lists[i:i+batch_size]
            batch_scores = self.model.forward(batch_questions, batch_contexts)
            scores.append(batch_scores)
        scores = torch.cat(scores, dim=0)
        if self.task_type == 'classification':
            scores = torch.argmax(scores, dim=-1)
            scores = torch.where(scores == 0, torch.ones_like(scores) * -1, scores)
        return scores.cpu().numpy().tolist()
    
    @torch.inference_mode()
    def get_test_mse(self, test_data, batch_size=64):
        self.eval()
        questions, context_lists, labels = test_data
        test_dataset = SelectorDataset(questions, context_lists, labels, self.task_type)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
        criterion = nn.MSELoss() if self.task_type == 'regression' else nn.CrossEntropyLoss()
        sign_criterion = SignLoss()
        total_loss = 0
        total_sign_loss = 0
        for batch in test_dataloader:
            batch_questions = batch['questions']
            batch_contexts = batch['contexts']
            batch_scores = self.model.forward(batch_questions, batch_contexts)
            loss = criterion(batch_scores, batch['labels'].to(self.device))
            total_loss += loss.item()
            total_sign_loss += sign_criterion(batch_scores, batch['labels'].to(self.device)).item()
        
        self.plot_loss_distribution(test_dataloader, nn.MSELoss(reduction='none'), 
                                    f'./visualization/{self.task_type}_loss_distribution_test.png')
        self.plot_loss_distribution(test_dataloader, SignLoss(reduction='none'), 
                                    f'./visualization/{self.task_type}_sign_loss_distribution_test.png')

        return total_loss / len(test_dataloader), total_sign_loss / len(test_dataloader)
