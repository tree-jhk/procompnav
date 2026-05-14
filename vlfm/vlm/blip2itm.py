
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

from transformers import AutoProcessor, Blip2ForImageTextRetrieval
import transformers
try:
    from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
except: # due to version mismatch with old transformers version depandent VLMs
    from transformers.modeling_utils import (
    PreTrainedModel,
    # apply_chunking_to_forward, 
    # find_pruneable_heads_and_indices, 
    # prune_linear_layer, 
)
    from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer 
    transformers.modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    transformers.modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    transformers.modeling_utils.prune_linear_layer = prune_linear_layer

try:
    from lavis.models import load_model_and_preprocess
except ModuleNotFoundError:
    print("Could not import lavis. This is OK if you are only using the client.")


class BLIP2ITM:
    """BLIP-2 Image-Text Retrieval/Matching model (Transformers)."""

    def __init__(
        self,
        model_id: str = "Salesforce/blip2-itm-vit-g",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but GPU-only mode was requested.")

        self.device = device
        self.model_id = model_id

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.model.eval()

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Compute the cosine similarity between the image and the prompt.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            txt (str): The text to compare the image to.

        Returns:
            float: The cosine similarity between the image and the prompt.
        """
        pil_img = Image.fromarray(image)
        inputs = self.processor(images=pil_img, text=txt, return_tensors="pt").to(self.device, torch.float16)
        out = self.model(**inputs, use_image_text_matching_head=False)
        cosine = out.logits_per_image[0, 0].float().item()
        return float(cosine)


class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        # print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument("--model_name", type=str, default="Salesforce/blip2-itm-vit-g")
    args = parser.parse_args()

    print("Loading model...")

    class BLIP2ITMServer(ServerMixin, BLIP2ITM):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {"response": self.cosine(image, payload["txt"])}

    blip = BLIP2ITMServer(model_id=args.model_name)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2itm", port=args.port)
