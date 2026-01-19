import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
from omegaconf import OmegaConf
from hydra import initialize, compose
import logging
from pathlib import Path
from proj.lightning_model import LightningAudioClassifier
from proj.data_module import AudioDataModule

log = logging.getLogger(__name__)


def train_lightning(
    model_cfg,
    train_cfg,
    max_epochs: int = 10,
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    log_dir: str = "logs",
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    """Train the model using PyTorch Lightning."""

    # Initialize the data module
    data_module = AudioDataModule(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=4,
        val_split=0.2
    )

    # Initialize the Lightning module
    model = LightningAudioClassifier(
        cfg=model_cfg,
        learning_rate=learning_rate
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="audio-classifier-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode="min"
    )

    # Setup logger
    logger = TensorBoardLogger(log_dir, name="audio_classifier")

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="auto",  # Automatically uses GPU if available
        devices="auto",      # Uses all available devices
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

    log.info("Training complete!")
    log.info(f"Best model saved at: {checkpoint_callback.best_model_path}")

    return trainer, model


def main():
    """Main training function."""
    with initialize(config_path="../../configs", version_base="1.1"):
        train_cfg = compose(config_name="train_cfg.yaml")
        model_cfg = compose(config_name="model_cfg.yaml")

    log.info("Training Configuration:")
    log.info(OmegaConf.to_yaml(train_cfg))
    log.info("\nModel Configuration:")
    log.info(OmegaConf.to_yaml(model_cfg))

    # Train with Lightning
    trainer, model = train_lightning(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        max_epochs=train_cfg.hyperparameters.epochs,
        data_dir=train_cfg.paths.data_dir,
        output_dir=train_cfg.paths.output_dir,
        batch_size=train_cfg.hyperparameters.batch_size,
        learning_rate=train_cfg.hyperparameters.get("learning_rate", 1e-3),
    )


if __name__ == "__main__":
    main()
