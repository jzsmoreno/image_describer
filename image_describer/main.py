import logging
import os
from io import BytesIO
from tempfile import NamedTemporaryFile

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TorchAoConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ImageDescriber:
    def __init__(self, model_path="models/", model_url="Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        Initialize the `ImageDescriber` class.

        Parameters
        ----------
        model_path: str
            Local path to save/load the model (default is `"models/"`)
        model_url: str
            HuggingFace model URL (default is `"Qwen/Qwen2.5-VL-3B-Instruct"`)
        """
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model on {self.device}")

        try:
            quantization_config = TorchAoConfig("int8_weight_only")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_url,
                cache_dir=model_path,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
            )

            if self.model.device != self.device:
                self.model.to(self.device)

            if torch.__version__ >= "2.0":
                self.model = torch.compile(self.model)
            self.processor = AutoProcessor.from_pretrained(
                model_url,
                model_type="qwen2_5_vl",
                cache_dir=model_path,
            )
        except Exception as e:
            print("Error:", e)

    def describe_image(self, image_input, max_new_tokens=128):
        """
        Describe the given image.

        Parameters
        ----------
        image_input: `str` or `BytesIO`
            Can be a local file path, URL, or BytesIO object

        Returns
        ----------
            Description of the image as a string
        """
        # Convert input to bytes if it's a BytesIO object
        if isinstance(image_input, BytesIO):
            img_bytes = image_input.getvalue()

            # Save to temporary file
            with NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                tmpfile.write(img_bytes)
                image_input = tmpfile.name

        messages = self._prompt(image_input, max_new_tokens=max_new_tokens)
        description = self._generate(messages=messages, max_new_tokens=max_new_tokens)
        messages = self._prompt(
            image_input,
            text="Returns a list of the elements in the image. Please describe each element in the image. Do not use additional comments.",
            max_new_tokens=max_new_tokens,
        )
        elements = self._generate(messages=messages, max_new_tokens=max_new_tokens)
        return self._format_description(description, elements)

    def _generate(self, messages, max_new_tokens):
        try:
            # Prepare inference inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move to device (CPU if no CUDA available)
            inputs = inputs.to(self.device)

            # Generate output
            with torch.no_grad():  # Avoid gradient computation for inference
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            return output_text[0]

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return ""

    def _format_description(self, description, elements):
        return f"\n\nVision Summary:\n\n" f"{description}\n\n" f"Insight:\n\n{elements}"

    def _prompt(self, image_input, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        text = kwargs.get("text", f"Describe this image. Use no more than {max_new_tokens} tokens.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            }
        ]

        return messages
