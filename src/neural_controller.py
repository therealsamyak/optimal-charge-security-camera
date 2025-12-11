import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralController(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )

        self.model_head = nn.Linear(64, 7)
        self.charge_head = nn.Linear(64, 1)

    def forward(self, x):
        shared_features = self.shared_layers(x)
        model_logits = self.model_head(shared_features)
        charge_logits = self.charge_head(shared_features)

        model_probs = F.softmax(model_logits, dim=-1)
        charge_prob = torch.sigmoid(charge_logits).squeeze(-1)

        return model_probs, charge_prob


class NeuralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        # Use weighted BCE to handle class imbalance
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(2.0)
        )  # Weight positive class more

    def forward(self, model_probs, charge_prob, model_targets, charge_targets):
        # Add small epsilon to prevent log(0)
        eps = 1e-8
        model_probs_clamped = torch.clamp(model_probs, min=eps, max=1 - eps)
        ce = self.ce_loss(model_probs_clamped.log(), model_targets)
        bce = self.bce_loss(charge_prob, charge_targets.float())
        return 0.5 * ce + 0.5 * bce
