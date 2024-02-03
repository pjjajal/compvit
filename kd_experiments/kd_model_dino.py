from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from dinov2.factory import dinov2_factory


class FeatureKD(nn.Module):
    def __init__(
        self,
        student_name: Literal[
            "dinov2_vittiny14",
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ],
        teacher_name: Literal[
            "dinov2_vittiny14",
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ],
        loss: Literal["l1-smooth", "l2"],
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.loss = loss

        self.student, self.student_cfg = dinov2_factory(student_name)
        self.teacher, self.teacher_cfg = dinov2_factory(teacher_name)

        dim_s = self.student_cfg.pop("embed_dim")
        dim_t = self.teacher_cfg.pop("embed_dim")

        # Projection layer from student feature dimension to teacher feature dimension.
        self.teacher_proj = nn.Linear(dim_s, dim_t)
        # self.flatten = nn.Flatten(1)

    def training_parameters(self):
        parameters = []
        parameters.extend(self.student.parameters())
        parameters.extend(self.teacher_proj.parameters())
        return parameters

    def forward_loss(self, h_teacher, h_student):
        if self.loss == "l1-smooth":
            return F.smooth_l1_loss(h_student, h_teacher, reduction="mean")
        elif self.loss == "l2":
            return F.mse_loss(h_teacher, h_student, reduction="mean")

    @torch.no_grad()
    def forward_teacher(self, x):
        h = self.teacher.forward_features(x)
        return h["x_norm"]

    def forward_student(self, x):
        h = self.student.forward_features(x)
        return h["x_norm"]

    def forward(self, x_teacher, x_student):
        # Get feature embeddings
        h_t = self.forward_teacher(x_teacher)
        h_s = self.forward_student(x_student)

        # Flatten embeddings
        # h_t = self.flatten(h_t)
        # h_s = self.flatten(h_s)

        # Project student features to teacher dimension.
        h_s = self.teacher_proj(h_s)
        # Compute loss
        loss = self.forward_loss(h_t, h_s)
        return loss
