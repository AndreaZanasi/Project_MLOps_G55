import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from hydra import initialize, compose
import logging
import wandb
from proj.lightning_model import LightningAudioClassifier
from proj.data_module import AudioDataModule

log = logging.getLogger(__name__)

def train_lightning(
    model_cfg,
    train_cfg,
    max_epochs: int = 10,
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    log_wandb: bool = True,
    accelerator: str = "auto",
    devices: str | int = "auto",
    data_module = None
):
    if data_module is None:
        data_module = AudioDataModule(
            data_dir=data_dir, output_dir=output_dir, batch_size=batch_size, num_workers=4, val_split=0.2
        )
    
    model = LightningAudioClassifier(
        cfg=model_cfg, 
        learning_rate=learning_rate
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=train_cfg.paths.model_dir,
        filename="audio-classifier-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    
    logger = None
    if log_wandb:
        logger = WandbLogger(
            entity="MLOps_G55",
            project="Project_MLOps_G55",
            config=OmegaConf.to_object(train_cfg)
        )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=10,
    )
    
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    
    if log_wandb and checkpoint_callback.best_model_path:
        best_model_path = checkpoint_callback.best_model_path
        artifact = wandb.Artifact(
            name="species_recognition_model",
            type="model",
            description="A model trained to recognize species based on animal vocalizations",
            metadata={"accuracy": float(checkpoint_callback.best_model_score)}
        )
        artifact.add_file(best_model_path, name="model.ckpt")
        wandb.log_artifact(artifact)
        wandb.finish()
    
    return trainer, model

def main():
    with initialize(config_path="../../configs", version_base="1.1"):
        train_cfg = compose(config_name="hydra_cfg.yaml")
        model_cfg = compose(config_name="model_cfg.yaml")

    log.info(OmegaConf.to_yaml(train_cfg))

    train_lightning(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        max_epochs=train_cfg.hyperparameters.epochs,
        data_dir=train_cfg.paths.data_dir,
        output_dir=train_cfg.paths.output_dir,
        batch_size=train_cfg.hyperparameters.batch_size,
        learning_rate=train_cfg.hyperparameters.get("learning_rate", 1e-3),
        log_wandb=train_cfg.logging.log_wandb
    )

if __name__ == "__main__":
    main()