import torch
from omegaconf import OmegaConf
from astra.pipelines.train_builder import PipelineBuilder

class InferenceModelBuilder:
    @staticmethod
    def build(config_dict: dict, ckpt_path: str):
        """
        Builds the PyTorch model and DataModule, and injects the trained weights.
        """
        # 1. Convert standard dict to OmegaConf (PipelineBuilder expects this)
        cfg = OmegaConf.create(config_dict)
        
        # 2. Build the "empty house" using your existing logic
        print("Initializing PipelineBuilder for inference...")
        builder = PipelineBuilder(cfg)
        builder.build_featurizers()
        builder.build_datamodule()
        builder.build_model_architecture()
        
        pytorch_model = builder.model_architecture
        
        # 3. Load the checkpoint safely (bypassing PyTorch 2.6 security since we trust our local files)
        print(f"Loading weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict']
        
        # 4. Clean the state_dict keys
        # PyTorch Lightning prefixes inner module keys with "model." because 
        # in AstraModule.__init__ you wrote: self.model = model
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                # Remove ONLY the first occurrence of "model."
                clean_key = k.replace('model.', '', 1)
                clean_state_dict[clean_key] = v
                
        # 5. Inject the weights into the base PyTorch model
        pytorch_model.load_state_dict(clean_state_dict)
        pytorch_model.eval() # Set to evaluation mode (disables dropout, etc.)
        
        print("Model weights injected successfully.")
        return builder, pytorch_model