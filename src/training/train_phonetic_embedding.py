"""
Training script for Phonetic Embedding Model using PyTorch Lightning.

Supports:
- Multi-GPU training with DDP
- Masked Language Modeling objective
- Automatic checkpointing and logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.phonetic_embedding import PhoneticEmbeddingModel
from src.data.dataset import CognateDataset
from src.data.phonetic_converter import PhoneticConverter


class PhoneticEmbeddingTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training phonetic embeddings.
    """
    
    def __init__(
        self,
        model: PhoneticEmbeddingModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        mask_prob: float = 0.15,
        warmup_steps: int = 1000
    ):
        super().__init__()
        
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.mask_prob = mask_prob
        self.warmup_steps = warmup_steps
        
        # MLM prediction head
        self.mlm_head = nn.Linear(model.embedding_dim, 256)  # vocab_size
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def mask_tokens(self, inputs: torch.Tensor):
        """
        Prepare masked input and labels for MLM.
        """
        labels = inputs.clone()
        
        # Create random mask (15% of tokens)
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Don't mask padding tokens
        masked_indices[inputs == 0] = False
        
        # 80% of the time, replace with [MASK] token (255)
        # 10% of the time, replace with random token
        # 10% of the time, keep original
        rand = torch.rand(labels.shape)
        
        masked_inputs = inputs.clone()
        masked_inputs[masked_indices & (rand < 0.8)] = 255  # MASK token
        
        random_tokens = torch.randint(1, 255, labels.shape, dtype=torch.long, device=inputs.device)
        masked_inputs[masked_indices & (rand >= 0.8) & (rand < 0.9)] = random_tokens[masked_indices & (rand >= 0.8) & (rand < 0.9)]
        
        # Only compute loss on masked tokens
        labels[~masked_indices] = -100
        
        return masked_inputs, labels
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for MLM.
        """
        # batch contains word pairs from CognateDataset
        # Use both words for MLM training
        word1 = batch['word1']
        word2 = batch['word2']
        
        # Combine both words
        inputs = torch.cat([word1, word2], dim=0)
        
        # Mask tokens
        masked_inputs, labels = self.mask_tokens(inputs)
        
        # Get sequence embeddings
        seq_emb = self.model.get_sequence_embeddings(masked_inputs)
        
        # Predict masked tokens
        predictions = self.mlm_head(seq_emb)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            predictions.view(-1, 256),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        word1 = batch['word1']
        word2 = batch['word2']
        
        inputs = torch.cat([word1, word2], dim=0)
        masked_inputs, labels = self.mask_tokens(inputs)
        
        seq_emb = self.model.get_sequence_embeddings(masked_inputs)
        predictions = self.mlm_head(seq_emb)
        
        loss = nn.functional.cross_entropy(
            predictions.view(-1, 256),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Calculate accuracy on masked tokens
        pred_tokens = predictions.argmax(dim=-1)
        mask = (labels != -100)
        accuracy = (pred_tokens[mask] == labels[mask]).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer and scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warmup
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                0.5 * (1.0 + torch.cos(torch.tensor((current_step - self.warmup_steps) / (self.trainer.max_steps - self.warmup_steps) * 3.14159)))
            )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


def train_phonetic_embedding(
    config_path: str = "configs/model_config.yaml",
    data_path: str = "data/raw/sample_etymology_data.json",
    output_dir: str = "outputs/phonetic_embedding",
    **kwargs
):
    """
    Main training function.
    
    Args:
        config_path: Path to model configuration
        data_path: Path to training data
        output_dir: Output directory for checkpoints
        **kwargs: Override config parameters
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['phonetic_embedding']
    training_config = config['training']
    
    # Override with kwargs
    model_config.update(kwargs.get('model', {}))
    training_config.update(kwargs.get('training', {}))
    
    # Create model
    model = PhoneticEmbeddingModel(
        vocab_size=model_config['embedding']['vocab_size'],
        embedding_dim=model_config['embedding']['embedding_dim'],
        num_layers=model_config['transformer']['num_layers'],
        num_heads=model_config['transformer']['num_heads'],
        d_ff=model_config['transformer']['d_ff'],
        dropout=model_config['transformer']['dropout'],
        max_seq_length=model_config['embedding']['max_sequence_length'],
        pooling=model_config['output']['pooling']
    )
    
    # Create datasets
    converter = PhoneticConverter()
    
    train_dataset = CognateDataset(
        data_path=data_path,
        phonetic_converter=converter,
        mode="pair"
    )
    
    val_dataset = CognateDataset(
        data_path=data_path,
        phonetic_converter=converter,
        mode="pair"
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create Lightning module
    pl_model = PhoneticEmbeddingTrainer(
        model=model,
        learning_rate=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay'])
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        filename='phonetic-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=training_config['early_stopping']['patience'],
        mode='min'
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='phonetic_embedding'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=training_config['max_epochs'],
        accelerator=training_config['accelerator'],
        devices=training_config['devices'],
        strategy=training_config['strategy'],
        precision=training_config['precision'],
        gradient_clip_val=training_config['gradient_clip_val'],
        accumulate_grad_batches=training_config['accumulate_grad_batches'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=50
    )
    
    # Train
    trainer.fit(pl_model, train_loader, val_loader)
    
    print(f"Training completed! Best model saved to {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Phonetic Embedding Model")
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--data', type=str, default='data/raw/sample_etymology_data.json')
    parser.add_argument('--output', type=str, default='outputs/phonetic_embedding')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--devices', type=int, default=None)
    
    args = parser.parse_args()
    
    # Build kwargs from args
    overrides = {}
    if args.epochs:
        overrides.setdefault('training', {})['max_epochs'] = args.epochs
    if args.batch_size:
        overrides.setdefault('training', {})['batch_size'] = args.batch_size
    if args.lr:
        overrides.setdefault('training', {})['learning_rate'] = args.lr
    if args.devices:
        overrides.setdefault('training', {})['devices'] = args.devices
    
    train_phonetic_embedding(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        **overrides
    )
