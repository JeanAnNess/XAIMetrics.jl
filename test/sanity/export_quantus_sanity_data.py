import os
import torch
import numpy as np
import torchvision.models as models
import sys
import quantus

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Repository root: {REPO_ROOT}")
QUANTUS_ASSETS = os.path.join(REPO_ROOT, "Quantus", "tutorials", "assets", "imagenet_samples")
OUTPUT_FILE = os.path.join(REPO_ROOT, "test", "assets", "quantus_intermediate.npz")

def main():
    sys.path.append(os.path.join(REPO_ROOT, "Quantus"))
    print(f"Loading assets from {QUANTUS_ASSETS}...")
    
    # 1. tutorial assets
    x_batch = torch.load(os.path.join(QUANTUS_ASSETS, "x_batch.pt"))
    y_batch = torch.load(os.path.join(QUANTUS_ASSETS, "y_batch.pt"))
    
    x_batch = x_batch[:10]  # Keep only first 10 samples for sanity check
    y_batch = y_batch[:10]
    
    print(f"x_batch shape: {x_batch.shape}")
    print(f"y_batch shape: {y_batch.shape}")

    # 2. ResNet18
    print("Loading ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    # 3. predictions
    print("Generating predictions...")
    with torch.no_grad():
        logits = model(x_batch)
        preds = torch.argmax(logits, dim=1)

    # 4. metrics
    print("Computing Quantus metrics (Baseline: InputXGradient)...")
    explain_kwargs = {
        "method": "InputXGradient",
        "xai_lib": "captum",
    }

    # avg sensitivity
    metric_as = quantus.AvgSensitivity(
        nr_samples=10,
        lower_bound=0.2,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        similarity_func=quantus.similarity_func.difference,
    )
    print(f"AvgSensitivity default perturb_func: {metric_as.perturb_func}")
    scores_as = metric_as(model=model, x_batch=x_batch.numpy(), y_batch=y_batch.numpy(), explain_func=quantus.explain, explain_func_kwargs=explain_kwargs)
    print(f"Avg Sensitivity scores: {scores_as[:3]}...")

    # local lipschitz estimate
    metric_lle = quantus.LocalLipschitzEstimate(
        nr_samples=10,
        perturb_std=0.2,
        perturb_mean=0.0,
        norm_numerator=quantus.similarity_func.distance_euclidean,
        norm_denominator=quantus.similarity_func.distance_euclidean,    
        similarity_func=quantus.similarity_func.lipschitz_constant,
    )
    print(f"LocalLipschitzEstimate default perturb_func: {metric_lle.perturb_func}")
    scores_lle = metric_lle(model=model, x_batch=x_batch.numpy(), y_batch=y_batch.numpy(), explain_func=quantus.explain, explain_func_kwargs=explain_kwargs)
    print(f"Local Lipschitz scores: {scores_lle[:3]}...")

    # 5. export
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    np.savez(
        OUTPUT_FILE,
        x_batch=x_batch.numpy(),
        y_batch=y_batch.numpy(),
        py_logits=logits.numpy(),
        py_preds=preds.numpy(),
        quantus_avg_sensitivity=np.array(scores_as),
        quantus_local_lipschitz=np.array(scores_lle)
    )
    
    print(f"Exported intermediate data to {OUTPUT_FILE}")
    print("Next step: Run 'julia test/sanity/create_julia_sanity_fixtures.jl' to create the JLD2 artifact.")

if __name__ == "__main__":
    main()