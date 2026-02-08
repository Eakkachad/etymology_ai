"""
Training script for Siamese Network using triplet loss.
"""

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.phonetic_embedding import PhoneticEmbeddingModel
from src.models.siamese_network import SiameseNetwork, TripletLoss
from src.data.dataset import CognateDataset
from src.data.phonetic_converter import PhoneticConverter


class SiameseTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training Siamese network.
    """
    
    def __init__(
        self,
        model: SiameseNetwork,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        margin: float = 0.5
    ):
        super().__init__()
        
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        
        # Triplet loss
        self.criterion = TripletLoss(margin=margin, distance_metric="cosine")
        
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, anchor, positive, negative):
        return self.model.forward_triplet(anchor, positive, negative)
    
    def training_step(self, batch, batch_idx):
        anchor = batch['anchor']
        positive = batch['positive']
        negative = batch['negative']
        
        # Get embeddings
        anchor_emb, pos_emb, neg_emb = self.model.forward_triplet(anchor, positive, negative)
        
        # Compute loss
        loss = self.criterion(anchor_emb, pos_emb, neg_emb)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        anchor = batch['anchor']
        positive = batch['positive']
        negative = batch['negative']
        
        anchor_emb, pos_emb, neg_emb = self.model.forward_triplet(anchor, positive, negative)
        loss = self.criterion(anchor_emb, pos_emb, neg_emb)
        
        # Calculate accuracy (positive should be closer than negative)
        pos_dist = torch.norm(anchor_emb - pos_emb, p=2, dim=1)
        neg_dist = torch.norm(anchor_emb - neg_emb, p=2, dim=1)
        accuracy = (pos_dist < neg_dist).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def train_siamese(
    config_path: str = "configs/model_config.yaml",
    data_path: str = "data/raw/sample_etymology_data.json",
    encoder_checkpoint: str = None,
    output_dir: str = "outputs/siamese",
    **kwargs
):
    """
    Train Siamese network.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create or load encoder
    if encoder_checkpoint:
        encoder = PhoneticEmbeddingModel.load_from_checkpoint(encoder_checkpoint)
    else:
        model_config = config['phonetic_embedding']
        encoder = PhoneticEmbeddingModel(
            vocab_size=model_config['embedding']['vocab_size'],
            embedding_dim=model_config['embedding']['embedding_dim'],
            num_layers=model_config['transformer']['num_layers'],
            num_heads=model_config['transformer']['num_heads'],
            pooling=model_config['output']['pooling']
        )
    
    # Create Siamese network
    siamese_config = config['siamese_network']
    siamese = SiameseNetwork(
        encoder=encoder,
        embedding_dim=512,
        projection_dims=siamese_config['projection']['hidden_dims'],
        dropout=siamese_config['projection']['dropout'],
        similarity_metric=siamese_config['similarity']['metric']
    )
    
    # Dataset
    converter = PhoneticConverter()
    train_dataset = CognateDataset(data_path, converter, mode="triplet")
    val_dataset = CognateDataset(data_path, converter, mode="triplet")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    
    # Lightning module
    pl_model = SiameseTrainer(
        model=siamese,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        margin=siamese_config['loss']['margin']
    )
    
    # Callbacks & Logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        filename='siamese-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        save_top_k=3
    )
    
    logger = TensorBoardLogger(output_dir, name='siamese')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        strategy=config['training']['strategy'],
        callbacks=[checkpoint_callback],
        logger=logger
    )
    
    trainer.fit(pl_model, train_loader, val_loader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/model_config.yaml')
    parser.add_argument('--data', default='data/raw/sample_etymology_data.json')
    parser.add_argument('--encoder', default=None, help='Path to pretrained encoder checkpoint')
    parser.add_argument('--output', default='outputs/siamese')
    args = parser.parse_args()
    
    train_siamese(args.config, args.data, args.encoder, args.output)
