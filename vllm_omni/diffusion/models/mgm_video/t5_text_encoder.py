# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MGM-Video T5 text encoder with MGM-specific prompt preprocessing.

This module encapsulates all T5 text encoding logic that was previously
embedded in pipeline_mgm_video.py. It handles:
- AutoTokenizer / T5EncoderModel loading
- MGM-specific prompt cleaning (including intentional bugs for binary alignment)
- Tokenization + T5 forward pass with bf16 storage / fp32 compute
- Classifier-free guidance (CFG) negative prompt handling
- Lazy device movement (CPU -> NPU on first encode)
"""

from __future__ import annotations

import html as html_mod
import logging
import os
import re

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel

logger = logging.getLogger(__name__)


class T5TextEncoder(nn.Module):
    """T5 text encoder for MGM-Video with MGM-specific prompt preprocessing."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = True,
        keep_on_cpu: bool = False,
    ):
        """Initialize T5 text encoder.

        Args:
            model_path: Path to the model directory (contains tokenizer/ and text_encoder/ subdirs).
            device: Target device (e.g. torch.device("npu:0")).
            dtype: Data type for T5 weights storage (default bf16).
            local_files_only: Whether to only load from local files.
            keep_on_cpu: If True, keep weights on CPU for lazy device movement.
                         Used in HSDP/SP/offload mode to save NPU memory during init.
        """
        super().__init__()
        self._device = device
        self._dtype = dtype

        tokenizer_path = os.path.join(model_path, "tokenizer") if local_files_only else model_path
        text_encoder_path = os.path.join(model_path, "text_encoder") if local_files_only else model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=local_files_only
        )
        # T5 encoder: store weights in bf16, compute in fp32 via explicit
        # .float()/.bfloat16() cast around the forward pass. This halves the
        # NPU memory footprint while preserving fp32 precision.
        self.text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_path, torch_dtype=dtype, local_files_only=local_files_only
        )
        if not keep_on_cpu:
            self.text_encoder = self.text_encoder.to(device)

    @property
    def device(self) -> torch.device:
        return self.text_encoder.device

    @staticmethod
    def _prompt_clean(text: str) -> str:
        """Clean text prompt to match MGM T5Embedder.clean_caption.

        The original MGM-Video repo applies aggressive text preprocessing
        before tokenization: lowercasing, URL/HTML removal, special character
        normalization, etc. Missing .lower() alone causes different token IDs
        and completely different T5 embeddings.

        NOTE: There is an intentional bug on line where text.strip() is called
        but its return value is discarded (matching the original repo). This bug
        affects which subsequent regexes match and is required for binary
        alignment with the original T5 embeddings.
        """
        import urllib.parse as ul

        text = str(text)
        text = ul.unquote_plus(text)
        text = text.strip().lower()

        # Remove <person> tags
        text = re.sub("<person>", "person", text)

        # Remove URLs
        text = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            "",
            text,
        )
        text = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            "",
            text,
        )

        # Remove HTML
        try:
            from bs4 import BeautifulSoup
            text = BeautifulSoup(text, features="html.parser").text
        except ImportError:
            text = re.sub(r"<[^>]+>", "", text)

        # Remove @mentions
        text = re.sub(r"@[\w\d]+\b", "", text)

        # Remove CJK characters
        text = re.sub(r"[㇀-㇯]+", "", text)
        text = re.sub(r"[ㇰ-ㇿ]+", "", text)
        text = re.sub(r"[㈀-㋿]+", "", text)
        text = re.sub(r"[㌀-㏿]+", "", text)
        text = re.sub(r"[㐀-䶿]+", "", text)
        text = re.sub(r"[䷀-䷿]+", "", text)
        text = re.sub(r"[一-鿿]+", "", text)

        # Normalize dashes and quotes
        text = re.sub(
            r"[-֊־᐀᠆‐-―⸗⸚⸺⸻⹀〜〰゠︱︲﹘﹣－]+",
            "-",
            text,
        )
        text = re.sub(r"[`´«»""¨]", '"', text)
        text = re.sub(r"[‘’]", "'", text)

        # Remove HTML entities
        text = re.sub(r"&quot;?", "", text)
        text = re.sub(r"&amp", "", text)

        # Remove IP addresses
        text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", text)

        # Remove article IDs
        text = re.sub(r"\d:\d\d\s+$", "", text)

        # Remove \\n
        text = re.sub(r"\\n", " ", text)

        # Remove #numbers
        text = re.sub(r"#\d{1,3}\b", "", text)
        text = re.sub(r"#\d{5,}\b", "", text)
        text = re.sub(r"\b\d{6,}\b", "", text)

        # Remove filenames
        text = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", text)

        # Normalize quotes and dots
        text = re.sub(r"[\"\']{2,}", r'"', text)
        text = re.sub(r"[\.]{2,}", r" ", text)

        # Remove bad punctuation
        bad_punct_regex = re.compile(
            r"["
            + "#®•©™&@·º½¾¿¡§~"
            + r"\)"
            + r"\("
            + r"\]"
            + r"\["
            + r"\}"
            + r"\{"
            + r"\|"
            + "\\"
            + r"\/"
            + r"\*"
            + "]{1,}"
        )
        text = re.sub(bad_punct_regex, r" ", text)
        text = re.sub(r"\s+\.\s+", r" ", text)

        # Normalize hyphens/underscores
        regex2 = re.compile(r"(?:\-|_)")
        if len(re.findall(regex2, text)) > 3:
            text = re.sub(regex2, " ", text)

        # Basic clean (ftfy + html unescape)
        try:
            import ftfy
            text = ftfy.fix_text(text)
        except ImportError:
            pass
        text = html_mod.unescape(html_mod.unescape(text))
        text = text.strip()

        # Remove alphanumeric codes
        text = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", text)
        text = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", text)
        text = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", text)

        # Remove common spam phrases
        text = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", text)
        text = re.sub(r"(free\s)?download(\sfree)?", "", text)
        text = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", text)
        text = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", text)
        text = re.sub(r"\bpage\s+\d+\b", "", text)

        text = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", text)
        text = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", text)

        # Normalize whitespace around colons and punctuation
        text = re.sub(r"\b\s+\:\s+", r": ", text)
        text = re.sub(r"(\D[,\./])\b", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing special chars
        # NOTE: intentionally NOT assigning the result back to match the
        # original MGM-Video-Ascend repo's clean_caption bug where
        # caption.strip() discards the return value. This bug affects
        # which subsequent regexes match, so preserving it is required
        # for binary alignment with the original T5 embeddings.
        text.strip()
        text = re.sub(r"^['\"]([\w\W]+)['\"]$", r"\1", text)
        text = re.sub(r"^[\'\_,\-\:;]", r"", text)
        text = re.sub(r"[\'\_,\-\:\-\+]$", r"", text)
        text = re.sub(r"^\.\S+$", "", text)

        return text.strip()

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 400,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Encode text prompt using T5 text encoder.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for CFG
            do_classifier_free_guidance: Whether to encode negative prompt
            num_videos_per_prompt: Number of videos per prompt
            max_sequence_length: Maximum sequence length for tokenizer
            device: Device to put embeddings on
            dtype: Data type for embeddings

        Returns:
            Tuple of (prompt_embeds, negative_prompt_embeds, attention_mask)
        """
        device = device or self._device
        dtype = dtype or self.text_encoder.dtype

        # Lazy move text_encoder to device (kept on CPU in HSDP mode to save memory)
        if self.text_encoder.device.type == "cpu" and device is not None:
            self.text_encoder = self.text_encoder.to(device)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(self._prompt_clean(p)) for p in prompt]
        batch_size = len(prompt_clean)

        # Tokenize
        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # Encode — T5 weights are bf16 but compute in fp32.
        # We explicitly cast the model to fp32 before the forward pass and
        # back to bf16 afterwards, matching the original repo's FSDP behavior
        # (bf16 storage, fp32 compute). Using torch.autocast(dtype=float32)
        # is NOT equivalent because autocast has an op whitelist/blacklist.
        with torch.no_grad():
            self.text_encoder.float()
            prompt_embeds = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds.last_hidden_state  # [B, L, D]
            self.text_encoder.bfloat16()

        # Reshape to [B, 1, L, D] to match MGM expected format
        prompt_embeds = prompt_embeds.unsqueeze(1)

        # Get negative prompt embeddings
        if do_classifier_free_guidance and negative_prompt is not None:
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_prompt_clean = [self._prompt_clean(p) for p in negative_prompt]

            uncond_inputs = self.tokenizer(
                neg_prompt_clean,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_inputs.input_ids.to(device)
            uncond_attention_mask = uncond_inputs.attention_mask.to(device)

            with torch.no_grad():
                self.text_encoder.float()
                negative_prompt_embeds = self.text_encoder(
                    input_ids=uncond_input_ids,
                    attention_mask=uncond_attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds.last_hidden_state
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1)
                self.text_encoder.bfloat16()
        elif do_classifier_free_guidance:
            # Use empty string as negative prompt (MGM default)
            uncond_inputs = self.tokenizer(
                [" "] * batch_size,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_inputs.input_ids.to(device)
            uncond_attention_mask = uncond_inputs.attention_mask.to(device)

            with torch.no_grad():
                self.text_encoder.float()
                negative_prompt_embeds = self.text_encoder(
                    input_ids=uncond_input_ids,
                    attention_mask=uncond_attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds.last_hidden_state
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(1)
                self.text_encoder.bfloat16()
        else:
            negative_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds, attention_mask
