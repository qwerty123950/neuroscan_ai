"""
NeuroScan AI — Evaluation Metrics Module.

Computes confusion matrix, accuracy, macro-averaged and per-class precision,
recall, and F1-score for 4-class Brain Tumor MRI classification:
0: glioma, 1: meningioma, 2: notumor, 3: pituitary.

Guarantees robust zero-division handling (never yields NaN, Inf, or raises ZeroDivisionError).
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

DEFAULT_CLASS_NAMES: Tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
NUM_CLASSES = len(DEFAULT_CLASS_NAMES)


def compute_confusion_matrix(
    targets: Union[torch.Tensor, Sequence[int]],
    preds: Union[torch.Tensor, Sequence[int]],
    num_classes: int = NUM_CLASSES,
) -> List[List[int]]:
    """
    Compute an [num_classes x num_classes] confusion matrix.
    Rows correspond to ground-truth classes; columns correspond to predicted classes.

    Args:
        targets: 1D tensor or sequence of integer class labels.
        preds: 1D tensor/sequence of predicted classes, or 2D tensor of raw logits [N, C].
        num_classes: Number of target classes (default 4).

    Returns:
        num_classes x num_classes integer matrix.
    """
    if isinstance(targets, torch.Tensor):
        targets_list = targets.detach().cpu().long().view(-1).tolist()
    else:
        targets_list = [int(t) for t in targets]

    if isinstance(preds, torch.Tensor):
        preds_tensor = preds.detach().cpu()
        if preds_tensor.ndim > 1:
            preds_tensor = preds_tensor.argmax(dim=-1)
        preds_list = preds_tensor.long().view(-1).tolist()
    else:
        preds_list = [int(p) for p in preds]

    if len(targets_list) != len(preds_list):
        raise ValueError(
            f"Targets length ({len(targets_list)}) does not match preds length ({len(preds_list)})"
        )

    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(targets_list, preds_list):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t][p] += 1
        else:
            raise ValueError(
                f"Index out of bounds for num_classes={num_classes}: target={t}, pred={p}"
            )

    return matrix


def compute_metrics_from_confusion_matrix(
    matrix: Union[torch.Tensor, Sequence[Sequence[int]]],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate classification metrics from a confusion matrix.

    Handles zero predictions/samples safely:
    - precision = 0.0 when (TP + FP) == 0
    - recall = 0.0 when (TP + FN) == 0
    - f1 = 0.0 when (precision + recall) == 0.0
    - macro metrics averaged across all num_classes

    Args:
        matrix: [num_classes x num_classes] integer confusion matrix.
        class_names: Optional sequence of class names in index order.

    Returns:
        Dictionary containing accuracy, macro_f1, macro_precision, macro_recall,
        per_class breakdown, confusion_matrix, and total_samples.
    """
    if class_names is None:
        class_names = list(DEFAULT_CLASS_NAMES)
    else:
        class_names = list(class_names)

    num_classes = len(class_names)

    if isinstance(matrix, torch.Tensor):
        mat = matrix.detach().cpu().long().tolist()
    else:
        mat = [[int(cell) for cell in row] for row in matrix]

    if len(mat) != num_classes or any(len(row) != num_classes for row in mat):
        raise ValueError(
            f"Confusion matrix shape ({len(mat)}x{len(mat[0]) if mat else 0}) "
            f"does not match num_classes ({num_classes})"
        )

    total_samples = sum(sum(row) for row in mat)
    total_correct = sum(mat[i][i] for i in range(num_classes))
    accuracy = (float(total_correct) / float(total_samples)) if total_samples > 0 else 0.0

    per_class: Dict[str, Dict[str, Union[float, int]]] = {}
    f1_list: List[float] = []
    precision_list: List[float] = []
    recall_list: List[float] = []

    for c, name in enumerate(class_names):
        tp = mat[c][c]
        fp = sum(mat[i][c] for i in range(num_classes) if i != c)
        fn = sum(mat[c][j] for j in range(num_classes) if j != c)
        support = sum(mat[c][j] for j in range(num_classes))

        pred_total = tp + fp
        precision = float(tp) / float(pred_total) if pred_total > 0 else 0.0

        recall = float(tp) / float(support) if support > 0 else 0.0

        pr_sum = precision + recall
        f1 = (2.0 * precision * recall) / pr_sum if pr_sum > 0.0 else 0.0

        per_class[name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    macro_precision = (sum(precision_list) / float(num_classes)) if num_classes > 0 else 0.0
    macro_recall = (sum(recall_list) / float(num_classes)) if num_classes > 0 else 0.0
    macro_f1 = (sum(f1_list) / float(num_classes)) if num_classes > 0 else 0.0

    return {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "per_class": per_class,
        "confusion_matrix": mat,
        "total_samples": total_samples,
    }


def compute_classification_metrics(
    targets: Union[torch.Tensor, Sequence[int]],
    preds: Union[torch.Tensor, Sequence[int]],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function computing confusion matrix and classification metrics directly
    from targets and predictions.
    """
    if class_names is None:
        class_names = list(DEFAULT_CLASS_NAMES)
    matrix = compute_confusion_matrix(targets, preds, num_classes=len(class_names))
    return compute_metrics_from_confusion_matrix(matrix, class_names=class_names)


class MetricTracker:
    """
    Memory-efficient metric accumulator across batches.
    Accumulates scalar loss and predictions without retaining PyTorch autograd computation graphs.
    """

    def __init__(self, class_names: Optional[Sequence[str]] = None) -> None:
        self.class_names = list(class_names) if class_names is not None else list(DEFAULT_CLASS_NAMES)
        self.num_classes = len(self.class_names)
        self.total_loss = 0.0
        self.total_samples = 0
        self.confusion_matrix = [[0 for _ in range(self.num_classes)] for _ in range(self.num_classes)]

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.total_loss = 0.0
        self.total_samples = 0
        self.confusion_matrix = [[0 for _ in range(self.num_classes)] for _ in range(self.num_classes)]

    def update(
        self,
        loss: float,
        preds: Union[torch.Tensor, Sequence[int]],
        targets: Union[torch.Tensor, Sequence[int]],
        batch_size: Optional[int] = None,
    ) -> None:
        """Update tracker with batch loss and predictions."""
        if isinstance(targets, torch.Tensor):
            targets_list = targets.detach().cpu().long().view(-1).tolist()
        else:
            targets_list = [int(t) for t in targets]

        if isinstance(preds, torch.Tensor):
            preds_tensor = preds.detach().cpu()
            if preds_tensor.ndim > 1:
                preds_tensor = preds_tensor.argmax(dim=-1)
            preds_list = preds_tensor.long().view(-1).tolist()
        else:
            preds_list = [int(p) for p in preds]

        n = len(targets_list) if batch_size is None else batch_size
        self.total_loss += float(loss) * n
        self.total_samples += n

        for t, p in zip(targets_list, preds_list):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t][p] += 1

    def compute(self) -> Dict[str, Any]:
        """Compute all classification metrics and average loss."""
        metrics = compute_metrics_from_confusion_matrix(self.confusion_matrix, class_names=self.class_names)
        avg_loss = (self.total_loss / float(self.total_samples)) if self.total_samples > 0 else 0.0
        metrics["loss"] = round(avg_loss, 6)
        return metrics

