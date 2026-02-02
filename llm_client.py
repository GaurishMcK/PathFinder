import yaml
import streamlit as st
from openai import OpenAI


def load_openai_client_from_yaml(path: str = "openai_config.yaml") -> tuple[OpenAI, str]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    openai_cfg = cfg.get("openai", {})
    api_key = openai_cfg.get("api_key")
    base_url = openai_cfg.get("base_url")
    model = openai_cfg.get("model")

    if not api_key:
        raise ValueError("Missing openai.api_key in YAML")
    if not model:
        raise ValueError("Missing openai.model in YAML")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


@st.cache_resource
def get_openai() -> tuple[OpenAI, str]:
    return load_openai_client_from_yaml("openai_config.yaml")
