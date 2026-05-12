from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptFormatter:
    template_name: str
    system_prompt: str

    def format(self, instruction: str, user_input: str | None = None, label: str | None = None) -> str:
        if self.template_name not in {"llama", "llama_chat", "qwen"}:
            raise ValueError(
                f"Unsupported template_name={self.template_name!r}. "
                "Supported templates: 'llama', 'llama_chat', 'qwen'."
            )

        instruction = self.system_prompt

        # ── LLaMA 3.1 Chat Template ──────────────────────────────────────────
        if self.template_name in {"llama", "llama_chat"}:
            if user_input:
                prompt = (
                    "<|start_header_id|>system<|end_header_id|>\n\n"
                    f"{instruction}<|eot_id|>"
                    "<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_input}<|eot_id|>"
                )
                if label is not None:
                    prompt += (
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                        f"{label}<|eot_id|>"
                    )
            else:
                prompt = (
                    "<|start_header_id|>user<|end_header_id|>"
                    f"{instruction}<|eot_id|>"
                )
            return prompt

        # ── Qwen2.5 Chat Template ─────────────────────────────────────────────
        # tokenizer_config.json 의 chat_template 확인 결과:
        # <|im_start|>{role}\n{content}<|im_end|>\n 형식 사용
        if self.template_name == "qwen":
            if user_input:
                prompt = (
                    f"<|im_start|>system\n{instruction}<|im_end|>\n"
                    f"<|im_start|>user\n{user_input}<|im_end|>\n"
                )
                if label is not None:
                    prompt += (
                        f"<|im_start|>assistant\n{label}<|im_end|>\n"
                    )
            else:
                prompt = (
                    f"<|im_start|>system\n{instruction}<|im_end|>\n"
                    f"<|im_start|>user\n<|im_end|>\n"
                )
            return prompt