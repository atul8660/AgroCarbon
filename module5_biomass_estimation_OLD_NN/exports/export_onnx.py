import argparse
import torch
from omegaconf import OmegaConf
from models.model import BiomassEstimator

def export_onnx(config_path: str, checkpoint_path: str, output_path: str):
    cfg = OmegaConf.load(config_path)
    model = BiomassEstimator(cfg)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy inputs matching inference shapes
    dummy_img = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)
    dummy_handcrafted = torch.randn(1, cfg.features.num_handcrafted)
    dummy_species = torch.tensor([0], dtype=torch.long)
    dummy_health = torch.tensor([0], dtype=torch.long)
    
    print(f"Exporting ONNX model to {output_path}...")
    torch.onnx.export(
        model, 
        (dummy_img, dummy_handcrafted, dummy_species, dummy_health), 
        output_path, 
        input_names=["image", "handcrafted", "species", "health"],
        output_names=["biomass_kg"],
        opset_version=14,
        dynamic_axes={
            "image": {0: "batch_size"},
            "handcrafted": {0: "batch_size"},
            "species": {0: "batch_size"},
            "health": {0: "batch_size"},
            "biomass_kg": {0: "batch_size"}
        }
    )
    print("Export successful.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/model.onnx")
    args = parser.parse_args()
    
    export_onnx(args.config, args.checkpoint, args.output)
