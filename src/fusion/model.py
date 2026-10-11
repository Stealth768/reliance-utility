"""Late/gated fusion network architecture with learned quality-agnostic gate.

Design:
- Two MLP encoders (enc_A, enc_B)
- Learned quality-agnostic gate: sigmoid(linear(enc_B_repr) + bias) (gate is not given severity S)
- Fused logit = gate * enc_A_logit + (1 - gate) * enc_B_logit
- Null handling: when modality A is 'removed' (a_removed=True), enc_A input is
  replaced with a learned null token parameter.
"""

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    _BaseModule = nn.Module
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    _BaseModule = object


class LateFusionModel(_BaseModule):
    """Gated multimodal fusion classifier with learned quality-agnostic gating.

    Parameters
    ----------
    dim_a : int
        Feature dimension of modality A.
    dim_b : int
        Feature dimension of modality B.
    hidden_dim : int, default=64
        Hidden representation dimension for each encoder.
    num_classes : int, default=4
        Number of output classification classes.
    """

    def __init__(
        self,
        dim_a: int = 64,
        dim_b: int = 64,
        hidden_dim: int = 64,
        num_classes: int = 4,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to instantiate LateFusionModel.")
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Learned null token for modality A when removed
        self.null_token_a = nn.Parameter(torch.zeros(dim_a))

        # Encoders
        self.enc_a = nn.Sequential(
            nn.Linear(dim_a, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.enc_b = nn.Sequential(
            nn.Linear(dim_b, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Classifiers per modality
        self.head_a = nn.Linear(hidden_dim, num_classes)
        self.head_b = nn.Linear(hidden_dim, num_classes)

        # Quality-agnostic gate based on modality B representation
        self.gate_fc = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_a: "torch.Tensor",
        x_b: "torch.Tensor",
        a_removed: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Forward pass.

        Parameters
        ----------
        x_a : torch.Tensor of shape (batch, dim_a)
            Modality A input.
        x_b : torch.Tensor of shape (batch, dim_b)
            Modality B input.
        a_removed : torch.Tensor of shape (batch,) or bool, optional
            Boolean mask indicating whether modality A is removed.

        Returns
        -------
        logits : torch.Tensor of shape (batch, num_classes)
            Fused output logits.
        gate : torch.Tensor of shape (batch, 1)
            Gate weights on modality A.
        """
        batch_size = x_a.shape[0]

        # Apply null token if modality A is removed
        if a_removed is not None:
            if isinstance(a_removed, bool):
                mask = torch.full((batch_size, 1), a_removed, device=x_a.device, dtype=torch.bool)
            else:
                mask = a_removed.view(batch_size, 1)
            null_expanded = self.null_token_a.unsqueeze(0).expand(batch_size, -1)
            x_a_eff = torch.where(mask, null_expanded, x_a)
        else:
            x_a_eff = x_a

        h_a = self.enc_a(x_a_eff)
        h_b = self.enc_b(x_b)

        logits_a = self.head_a(h_a)
        logits_b = self.head_b(h_b)

        # Gate computation from B representation (quality-agnostic, no S given)
        gate = torch.sigmoid(self.gate_fc(h_b))

        fused_logits = gate * logits_a + (1.0 - gate) * logits_b
        return fused_logits, gate
