"""Moteur de rédaction : un seul objet `LLMClient` pour les backends transformers (GPU), openai (Ollama, LM Studio...) et anthropic."""
import gc

import requests

from .environment import free_gpu
from .utils import get_secret, quiet_hf_downloads


class LLMClient:
    def __init__(self, cfg, env):
        self.cfg, self.env = cfg, env
        self.mode = "none"
        self.tok = None
        self.model = None

    # ------------------------- serveurs openai-compatibles -------------------
    def openai_headers(self):
        key = self.cfg.LLM_API_KEY or get_secret("OPENAI_API_KEY") or ""
        return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}

    def openai_up(self):
        try:
            return requests.get(self.cfg.LLM_BASE_URL.rstrip("/") + "/models", headers=self.openai_headers(), timeout=3).ok
        except Exception:
            return False

    # ------------------------- modèle local transformers ---------------------
    def load_hf_llm(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from packaging.version import Version
        quiet_hf_downloads()          # téléchargement de ~15 Go : pas de barre (voir utils.quiet_hf_downloads)
        cfg, env = self.cfg, self.env
        torch = env.torch
        dtype_key = "dtype" if Version(transformers.__version__.split("+")[0]) >= Version("4.56") else "torch_dtype"
        print(f"Chargement de {cfg.LLM_HF_MODEL} ({'4 bits' if env.has_cuda and cfg.LLM_4BIT else 'précision réduite'}) ...")
        tok = AutoTokenizer.from_pretrained(cfg.LLM_HF_MODEL)
        kwargs = {"device_map": "auto"}
        if env.has_cuda and cfg.LLM_4BIT:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs[dtype_key] = torch.float16 if env.has_cuda else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_HF_MODEL, **kwargs)
        model.eval()
        return tok, model

    # ------------------------- cycle de vie ------------------------------------
    def init(self):
        cfg = self.cfg
        mode = cfg.LLM_BACKEND
        if mode == "auto":
            mode = "transformers" if self.env.has_cuda else ("openai" if self.openai_up() else "none")
        if mode == "openai" and not self.openai_up():
            print(f"Serveur LLM injoignable ({cfg.LLM_BASE_URL}) : repli sur le résumé extractif.")
            mode = "none"
        if mode == "anthropic" and not get_secret("ANTHROPIC_API_KEY"):
            print("Clé ANTHROPIC_API_KEY introuvable : repli sur le résumé extractif.")
            mode = "none"
        if mode == "transformers":
            try:
                self.tok, self.model = self.load_hf_llm()
            except Exception as e:
                print("Chargement du LLM impossible :", str(e)[:300], "\nRepli sur le résumé extractif.")
                mode = "none"
        self.mode = mode
        print("Moteur de rédaction :", {"none": "aucun (résumé extractif)"}.get(mode, mode))

    def free(self):
        self.tok = self.model = None
        gc.collect()
        free_gpu(self.env)

    # ------------------------- appel ---------------------------------------------
    def call(self, prompt, max_tokens=700):
        cfg = self.cfg
        if self.mode == "transformers":
            torch = self.env.torch
            tok, model = self.tok, self.model
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False)
            return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if self.mode == "openai":
            r = requests.post(cfg.LLM_BASE_URL.rstrip("/") + "/chat/completions", headers=self.openai_headers(), timeout=1200,
                              json={"model": cfg.LLM_MODEL, "temperature": 0.2, "max_tokens": max_tokens,
                                    "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        if self.mode == "anthropic":
            r = requests.post(cfg.ANTHROPIC_URL, timeout=600,
                              headers={"x-api-key": get_secret("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01",
                                       "content-type": "application/json"},
                              json={"model": cfg.ANTHROPIC_MODEL, "max_tokens": max_tokens, "temperature": 0.2,
                                    "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
        raise RuntimeError("Aucun LLM actif")
