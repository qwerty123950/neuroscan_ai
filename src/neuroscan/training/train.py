"""
NeuroScan AI — Model Training Pipeline.

Orchestrates reproducible model training and validation:
- Config-driven execution from YAML.
- Stratified 80/20 train/validation partitioning with clean Testing split quarantine.
- Seeded model initialization prior to DataLoader iteration.
- Model factory supporting PaperGenericCNN and BaselineCNN architectures.
- SGD / AdamW optimizers and CrossEntropyLoss.
- Epoch-level validation with MetricTracker (confusion matrix, safe zero-division macro-F1).
- Best checkpoint saving and early stopping governed by validation macro-F1.
- Machine-readable JSON metrics history exported to reports/metrics/.
- Memory-efficient CPU execution without computation graph retention.
"""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from neuroscan.data.config import DataConfig
from neuroscan.data.dataloader import ReproducibleDataLoader, seed_worker
from neuroscan.data.dataset import CLASS_TO_IDX, IDX_TO_CLASS, get_default_project_root
from neuroscan.data.split import create_train_val_datasets
from neuroscan.evaluation.metrics import DEFAULT_CLASS_NAMES, MetricTracker
from neuroscan.models import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuroscan.training")


def load_training_config(
    config_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load and validate experiment configuration from YAML.
    """
    root = Path(project_root).resolve() if project_root else get_default_project_root()
    if config_path is None:
        target_path = root / "configs" / "train_config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.is_file():
        raise FileNotFoundError(f"Training configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML content in {target_path}: expected dictionary root.")

    # Validate essential schema
    required_sections = ["experiment", "data", "model", "training", "validation", "logging"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section '{section}' in {target_path}")

    return config


def set_all_seeds(seed: int) -> None:
    """
    Set process-wide seeds for random and torch RNGs.
    """
    random.seed(seed)
    torch.manual_seed(seed)


class EarlyStopping:
    """
    Monitors a metric to stop training when performance plateaus.
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 0.001,
    ) -> None:
        self.patience = patience
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False
        self.best_epoch: int = 0

    def step(self, score: float, epoch: int) -> bool:
        """
        Evaluate score against best_score.
        Returns True if a new significant improvement was achieved, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True

        if self.mode == "max":
            improvement = score - self.best_score
        else:
            improvement = self.best_score - score

        if improvement >= self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_val_macro_f1: float,
    val_metrics: Dict[str, Any],
    seed: int,
    config: Dict[str, Any],
) -> None:
    """
    Persist training checkpoint with weights, states, and reproduction metadata.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_macro_f1": float(best_val_macro_f1),
        "val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
        "val_loss": float(val_metrics.get("loss", 0.0)),
        "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
        "val_metrics": val_metrics,
        "class_to_idx": dict(CLASS_TO_IDX),
        "idx_to_class": dict(IDX_TO_CLASS),
        "class_names": list(DEFAULT_CLASS_NAMES),
        "seed": seed,
        "config": config,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint_dict, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path} (epoch {epoch}, val_macro_f1: {best_val_macro_f1:.4f})")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tracker: MetricTracker,
) -> Dict[str, Any]:
    """
    Execute one full training epoch across batches.
    Memory-efficient CPU execution: detaches predictions and accumulates scalar loss.
    """
    model.train()
    tracker.reset()

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Detach and track without keeping autograd computational graph
        batch_loss = loss.item()
        preds = outputs.argmax(dim=-1).detach().cpu()
        targets_cpu = targets.detach().cpu()
        tracker.update(loss=batch_loss, preds=preds, targets=targets_cpu, batch_size=images.size(0))

    return tracker.compute()


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tracker: MetricTracker,
) -> Dict[str, Any]:
    """
    Execute validation pass over validation dataloader.
    """
    model.eval()
    tracker.reset()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_loss = loss.item()
            preds = outputs.argmax(dim=-1).detach().cpu()
            targets_cpu = targets.detach().cpu()
            tracker.update(loss=batch_loss, preds=preds, targets=targets_cpu, batch_size=images.size(0))

    return tracker.compute()


def run_training(
    config_path: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Main training execution function.

    If dry_run=True, validates pipeline construction, checks dataset quarantine,
    seeds model and optimizer, runs a single dummy batch forward check, and returns
    without executing training epochs or modifying disk state.
    """
    root = Path(project_root).resolve() if project_root else get_default_project_root()
    config = load_training_config(config_path=config_path, project_root=root)

    exp_cfg = config["experiment"]
    data_cfg_dict = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    val_cfg = config["validation"]
    log_cfg = config["logging"]

    seed = int(exp_cfg.get("seed", 42))
    device = torch.device(exp_cfg.get("device", "cpu"))

    logger.info(f"Initializing NeuroScan AI training pipeline (experiment: '{exp_cfg.get('name')}', seed: {seed}, device: {device})")

    # 1. Seed global RNG before model initialization
    set_all_seeds(seed)

    # 2. Instantiate Model via factory and initialize weights
    num_classes = int(model_cfg.get("num_classes", 4))
    in_channels = int(model_cfg.get("in_channels", 3))
    arch_name = str(model_cfg.get("architecture", "PaperGenericCNN"))
    model = get_model(
        architecture=arch_name,
        num_classes=num_classes,
        in_channels=in_channels,
        dropout=float(model_cfg.get("dropout", 0.5)),
        dense_units=model_cfg.get("dense_units", None),
    )
    model.to(device)
    param_counts = model.count_parameters()
    logger.info(f"{arch_name} instantiated with {param_counts['total']:,} total parameters ({param_counts['trainable']:,} trainable)")

    # 3. Construct Data Pipelines and Stratified Datasets
    val_ratio = float(data_cfg_dict.get("val_ratio", 0.2))
    img_size = int(data_cfg_dict.get("img_size", 256))
    normalize_mode = str(data_cfg_dict.get("normalize_mode", "rescale_only"))
    brightness_mode = str(data_cfg_dict.get("brightness_mode", "paper_literal"))
    data_config = DataConfig(
        img_size=img_size,
        batch_size=int(data_cfg_dict.get("batch_size", 16)),
        num_workers=int(data_cfg_dict.get("num_workers", 0)),
        pin_memory=bool(data_cfg_dict.get("pin_memory", False)),
        manifest_path=data_cfg_dict.get("manifest_path", "data/processed/dataset_manifest.csv"),
        seed=seed,
        normalize_mode=normalize_mode,
        brightness_mode=brightness_mode,
    )

    train_dataset, val_dataset = create_train_val_datasets(
        config=data_config,
        val_ratio=val_ratio,
        seed=seed,
        project_root=root,
    )

    # Strict Quarantine Assertions
    if train_dataset.split != "Training" or val_dataset.split != "Training":
        raise RuntimeError("Clean testing split leakage detected: train/val datasets must only draw from 'Training'.")

    logger.info(
        f"Datasets ready: train={len(train_dataset)} samples, val={len(val_dataset)} samples "
        f"(Testing split strictly quarantined with 0 samples loaded)"
    )

    # 4. Construct DataLoaders
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = ReproducibleDataLoader(
        dataset=train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        generator=train_generator,
        worker_init_fn=seed_worker if seed is not None else None,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )

    # 5. Optimizer, Loss, and Scheduler
    opt_cfg = train_cfg.get("optimizer", {})
    opt_name = str(opt_cfg.get("name", "SGD")).upper()
    lr = float(opt_cfg.get("learning_rate", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if opt_name == "SGD":
        momentum = float(opt_cfg.get("momentum", 0.0))
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif opt_name == "ADAM":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1e-8))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    elif opt_name == "ADAMW":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1e-8))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    else:
        raise ValueError(f"Unsupported optimizer '{opt_name}'. Use 'SGD', 'Adam', or 'AdamW'.")

    loss_cfg = train_cfg.get("loss", {})
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    sched_cfg = train_cfg.get("scheduler", {})
    sched_name = sched_cfg.get("name") if sched_cfg else None
    scheduler = None
    if sched_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_cfg.get("mode", "min"),
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 2)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )

    # 6. Early Stopping & Checkpoint Paths
    es_cfg = val_cfg.get("early_stopping", {})
    early_stopper = None
    if es_cfg.get("enabled", False):
        early_stopper = EarlyStopping(
            patience=int(es_cfg.get("patience", 5)),
            mode=str(es_cfg.get("mode", "max")),
            min_delta=float(es_cfg.get("min_delta", 0.001)),
        )

    checkpoint_dir = root / log_cfg.get("checkpoint_dir", "models/checkpoints")
    checkpoint_filename = log_cfg.get("checkpoint_filename", "paper_generic_cnn_best.pt")
    checkpoint_path = checkpoint_dir / checkpoint_filename

    metrics_dir = root / log_cfg.get("metrics_output_dir", "reports/metrics")
    metrics_summary_path = metrics_dir / "paper_generic_cnn_training_summary.json"

    # DRY RUN EXIT
    if dry_run:
        logger.info("Executing dry-run verification...")
        dummy_input = torch.randn(2, 3, data_config.img_size, data_config.img_size, device=device)
        model.eval()
        with torch.no_grad():
            dummy_out = model(dummy_input)
            dummy_loss = criterion(dummy_out, torch.tensor([0, 1], device=device))

        report = {
            "status": "success",
            "dry_run": True,
            "architecture": model.__class__.__name__,
            "num_parameters": param_counts["total"],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "testing_samples_accessed": 0,
            "testing_split_quarantined": True,
            "dummy_forward_shape": list(dummy_out.shape),
            "dummy_loss": float(dummy_loss.item()),
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__ if scheduler else None,
            "epochs_run": 0,
            "optimization_steps_run": 0,
            "reproducibility": {
                "model_seeded_before_dataloader_iter": True,
                "dataset_seed": seed,
            },
        }
        logger.info("Dry-run verification completed successfully: 0 optimization steps executed.")
        return report

    # 7. Training Loop Execution
    epochs = int(train_cfg.get("epochs", 10))
    best_val_macro_f1 = -1.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    train_tracker = MetricTracker(class_names=DEFAULT_CLASS_NAMES)
    val_tracker = MetricTracker(class_names=DEFAULT_CLASS_NAMES)

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            tracker=train_tracker,
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            tracker=val_tracker,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        val_macro_f1 = val_metrics["macro_f1"]
        val_loss = val_metrics["loss"]

        if scheduler is not None:
            if sched_cfg.get("mode", "min") == "min":
                scheduler.step(val_loss)
            else:
                scheduler.step(val_macro_f1)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_macro_f1=best_val_macro_f1,
                val_metrics=val_metrics,
                seed=seed,
                config=config,
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_per_class": train_metrics["per_class"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class": val_metrics["per_class"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "lr": current_lr,
        }
        history.append(epoch_record)

        if early_stopper is not None:
            early_stopper.step(score=val_macro_f1, epoch=epoch)
            if early_stopper.early_stop:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"val_macro_f1 did not improve by {early_stopper.min_delta} for {early_stopper.patience} epochs."
                )
                break

    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "experiment_name": exp_cfg.get("name"),
        "seed": seed,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "epochs_requested": epochs,
        "epochs_trained": len(history),
        "stopped_early": early_stopper.early_stop if early_stopper else False,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "class_names": list(DEFAULT_CLASS_NAMES),
        "clean_test_quarantined": True,
        "clean_test_samples_evaluated": 0,
        "history": history,
    }
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Training summary persisted: {metrics_summary_path}")

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroScan AI — Model Training")
    parser.add_argument("--config", type=str, default=None, help="Path to training config YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform configuration, dataset, model, and forward check without training",
    )
    args = parser.parse_args()

    run_training(config_path=args.config, dry_run=args.dry_run)
