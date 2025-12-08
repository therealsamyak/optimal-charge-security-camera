import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralController(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )

        self.model_head = nn.Linear(64, 6)
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
        self.bce_loss = nn.BCELoss()

    def forward(self, model_probs, charge_prob, model_targets, charge_targets):
        ce = self.ce_loss(model_probs.log(), model_targets)
        bce = self.bce_loss(charge_prob, charge_targets.float())
        return 0.5 * ce + 0.5 * bce
