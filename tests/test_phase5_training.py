"""Comprehensive Tests for Phase 5: Advanced Training Techniques."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSelfSupervisedPretraining:
    """Test Self-Supervised Pretraining."""

    def test_mae_import(self):
        """Test that MAE can be imported."""
        from ml.training.self_supervised_pretrain import MAE

        assert MAE is not None

    def test_mae_forward(self):
        """Test MAE loss computation."""
        from ml.training.self_supervised_pretrain import MAELoss

        # MAELoss is a loss function, not a model.
        mae_loss = MAELoss(patch_size=16)

        # Create dummy reconstructed patches, target patches, and mask.
        B, N = 2, 196  # 14x14 patches.
        P = 16
        C = 3
        patch_dim = P * P * C

        recon = torch.randn(B, N, patch_dim)
        target = torch.randn(B, N, patch_dim)
        mask = torch.rand(B, N) > 0.25  # 75% masked.

        loss = mae_loss(recon, target, mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar.
        assert loss.item() >= 0

    def test_simclr_import(self):
        """Test that SimCLR can be imported."""
        from ml.training.self_supervised_pretrain import SimCLR

        assert SimCLR is not None

    def test_simclr_forward(self):
        """Test SimCLR loss computation."""
        from ml.training.self_supervised_pretrain import SimCLRLoss

        # SimCLRLoss is a loss function, not a model.
        simclr_loss = SimCLRLoss(temperature=0.07)

        # Create dummy normalized embeddings from two augmented views.
        B, D = 4, 256
        z1 = torch.randn(B, D)  # Embeddings from view 1.
        z2 = torch.randn(B, D)  # Embeddings from view 2.

        loss = simclr_loss(z1, z2)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar.
        assert loss.item() >= 0


class TestKnowledgeDistillation:
    """Test Knowledge Distillation."""

    def test_knowledge_distillation_import(self):
        """Test that Knowledge Distillation can be imported."""
        from ml.training.self_supervised_pretrain import KnowledgeDistillation

        assert KnowledgeDistillation is not None

    def test_knowledge_distillation_loss(self):
        """Test Knowledge Distillation loss computation."""
        from ml.training.self_supervised_pretrain import KnowledgeDistillationLoss

        # KnowledgeDistillationLoss is a loss function, not a model.
        kd_loss = KnowledgeDistillationLoss(temperature=3.0, alpha=0.7)

        # Create dummy logits.
        B, C = 4, 10
        student_logits = torch.randn(B, C)
        teacher_logits = torch.randn(B, C)
        labels = torch.randint(0, C, (B,))

        loss_dict = kd_loss(student_logits, teacher_logits, labels)

        assert isinstance(loss_dict, dict)
        assert "total_loss" in loss_dict
        assert "kd_loss" in loss_dict
        assert "ce_loss" in loss_dict
        assert loss_dict["total_loss"].item() >= 0


class TestContinualLearning:
    """Test Continual Learning (EWC)."""

    def test_ewc_import(self):
        """Test that EWC can be imported."""
        from ml.training.self_supervised_pretrain import ElasticWeightConsolidation

        assert ElasticWeightConsolidation is not None

    def test_ewc_initialization(self):
        """Test EWC initialization."""
        from ml.training.self_supervised_pretrain import ElasticWeightConsolidation

        model = nn.Linear(10, 10)
        ewc = ElasticWeightConsolidation(model, lambda_ewc=0.4)

        assert ewc is not None
        assert ewc.lambda_ewc == 0.4

    def test_ewc_loss(self):
        """Test EWC loss computation."""
        from ml.training.self_supervised_pretrain import ElasticWeightConsolidation

        model = nn.Linear(10, 10)
        ewc = ElasticWeightConsolidation(model, lambda_ewc=0.4)

        for name, param in model.named_parameters():
            ewc.fisher[name] = torch.ones_like(param.data)
            ewc.optimal_params[name] = param.data.clone()

        # Compute EWC penalty.
        ewc_penalty = ewc.penalty()

        assert ewc_penalty is not None
        assert isinstance(ewc_penalty, torch.Tensor)
        assert ewc_penalty.item() >= 0


class TestDataAugmentation:
    """Test Data Augmentation."""

    def test_multi_modal_augment_import(self):
        """Test that Multi-Modal Augmentation can be imported."""
        from ml.data.multi_modal_augment import MultiModalAugmentation

        assert MultiModalAugmentation is not None

    def test_multi_modal_augment_forward(self):
        """Test Multi-Modal Augmentation forward pass."""
        from ml.data.multi_modal_augment import MultiModalAugmentation

        augment = MultiModalAugmentation()

        # One float image in [0, 1] and MFCC audio [1, 13, T], as ml/data/dataset.py produces. This used
        # to skip on any exception: the 1-D audio hid that RandomEqualize rejects float images (~50% of calls).
        for seed in range(20):
            torch.manual_seed(seed)
            image = torch.rand(3, 224, 224)
            audio = torch.randn(1, 13, 100)
            aug_image, aug_audio = augment(image, audio)
            assert aug_image.shape == image.shape and aug_image.dtype == torch.float32
            assert 0.0 <= float(aug_image.min()) and float(aug_image.max()) <= 1.0
            assert aug_audio.shape == audio.shape

    def test_multi_modal_augment_keeps_uint8_images_uint8(self):
        from ml.data.multi_modal_augment import MultiModalAugmentation

        image = (torch.rand(3, 64, 64) * 255).to(torch.uint8)
        aug_image, aug_audio = MultiModalAugmentation()(image)
        assert aug_image.dtype == torch.uint8 and aug_image.shape == image.shape and aug_audio is None

    @pytest.mark.parametrize(
        "image,audio",
        [(torch.randn(3, 32, 32), None), (torch.rand(3, 32, 32), torch.randn(128))],
        ids=["normalised_image", "raw_waveform"],
    )
    def test_multi_modal_augment_refuses_inputs_it_would_corrupt(self, image, audio):
        from ml.data.multi_modal_augment import MultiModalAugmentation

        with pytest.raises(ValueError):
            MultiModalAugmentation()(image, audio)

    def test_synthetic_scene_generator_import(self):
        """Test that Synthetic Scene Generator can be imported."""
        from ml.data.synthetic_scene_generator import SyntheticSceneGenerator

        assert SyntheticSceneGenerator is not None


class TestCrossViewTraining:
    """Test Cross-View Training."""

    def test_cross_view_import(self):
        """Test that Cross-View Training can be imported."""
        from ml.retrieval.cross_view.cv_training import CrossViewTrainer

        assert CrossViewTrainer is not None

    def test_cross_view_forward(self):
        """Test Cross-View Training forward pass."""
        from ml.retrieval.cross_view.cv_training import CrossViewTrainer

        cv_training = CrossViewTrainer(embed_dim=256, temperature=0.07)
        cv_training.eval()

        view1 = torch.randn(2, 256)
        view2 = torch.randn(2, 256)
        negatives = torch.randn(5, 256)

        loss = cv_training.contrastive_loss(view1, view2, negatives)

        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss).all() and float(loss) >= 0.0


def run_all_tests():
    """Run all Phase 5 tests."""
    print("=" * 60)
    print("Phase 5: Advanced Training Techniques Tests")
    print("=" * 60)

    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()
