from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2:
    def __init__(
        self,
        model_id: str = "Salesforce/blip2-flan-t5-xl",
        device: Optional[Any] = None,
    ) -> None:
        # GPU-only
        if device is None:
            device = torch.device("cuda")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but GPU-only mode was requested.")

        self.device = device
        self.model_id = model_id

        self.processor = Blip2Processor.from_pretrained(model_id)

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
        ).to(self.device)

        self.model.eval()


    def ask(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        """Generates a caption for the given image.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            prompt (str, optional): An optional prompt to provide context and guide
                the caption generation. Can be used to ask questions about the image.

        Returns:
            dict: The generated caption.

        """
        pil_img = Image.fromarray(image)
        # with torch.inference_mode():
        #     processed_image = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        #     if prompt is None or prompt == "":
        #         out = self.model.generate({"image": processed_image})[0]
        #     else:
        #         out = self.model.generate({"image": processed_image, "prompt": prompt})[0]
        inputs = self.processor(pil_img, prompt, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs)
        out = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        return out


class BLIP2Client:
    def __init__(self, port: int = 12185):
        self.url = f"http://localhost:{port}/blip2"

    def ask(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        if prompt is None:
            prompt = ""
        response = send_request(self.url, image=image, prompt=prompt)

        return response["response"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--model_id", type=str, default="Salesforce/blip2-flan-t5-xl")
    args = parser.parse_args()

    print("Loading model...")

    class BLIP2Server(ServerMixin, BLIP2):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {"response": self.ask(image, payload.get("prompt"))}

    blip = BLIP2Server(model_id=args.model_id)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2", port=args.port)
