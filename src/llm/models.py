import os
import json
import logging
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_xai import ChatXAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_gigachat import GigaChat
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from enum import Enum
from pydantic import BaseModel, SecretStr
from typing import Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""

    ALIBABA = "Alibaba"
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GOOGLE = "Google"
    GROQ = "Groq"
    META = "Meta"
    MISTRAL = "Mistral"
    OPENAI = "OpenAI"
    OPENAI_COMPATIBLE = "OpenAI Compatible"
    OLLAMA = "Ollama"
    OPENROUTER = "OpenRouter"
    GIGACHAT = "GigaChat"
    AZURE_OPENAI = "Azure OpenAI"
    XAI = "xAI"
    DASHSCOPE = "Dashscope"
    IDEALAB = "IDEALAB"


class LLMModel(BaseModel):
    """Represents an LLM model configuration"""

    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def is_custom(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name == "-"

    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        if self.is_deepseek() or self.is_gemini():
            return False
        # Only certain Ollama models support JSON mode
        if self.is_ollama():
            return "llama3" in self.model_name or "neural-chat" in self.model_name
        # OpenRouter models generally support JSON mode
        if self.provider == ModelProvider.OPENROUTER:
            return True
        return True

    def is_deepseek(self) -> bool:
        """Check if the model is a DeepSeek model"""
        return self.model_name.startswith("deepseek")

    def is_gemini(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name.startswith("gemini")

    def is_ollama(self) -> bool:
        """Check if the model is an Ollama model"""
        return self.provider == ModelProvider.OLLAMA


# Load models from JSON file
def load_models_from_json(json_path: str) -> List[LLMModel]:
    """Load models from a JSON file"""
    with open(json_path, 'r') as f:
        models_data = json.load(f)
    
    models = []
    for model_data in models_data:
        # Convert string provider to ModelProvider enum
        provider_enum = ModelProvider(model_data["provider"])
        models.append(
            LLMModel(
                display_name=model_data["display_name"],
                model_name=model_data["model_name"],
                provider=provider_enum
            )
        )
    return models


# Get the path to the JSON files
current_dir = Path(__file__).parent
models_json_path = current_dir / "api_models.json"
ollama_models_json_path = current_dir / "ollama_models.json"

# Load available models from JSON
AVAILABLE_MODELS = load_models_from_json(str(models_json_path))

# Load Ollama models from JSON
OLLAMA_MODELS = load_models_from_json(str(ollama_models_json_path))

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

# Create Ollama LLM_ORDER separately
OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


def get_model_info(model_name: str, model_provider: str) -> LLMModel | None:
    """Get model information by model_name"""
    all_models = AVAILABLE_MODELS + OLLAMA_MODELS
    return next((model for model in all_models if model.model_name == model_name and model.provider == model_provider), None)


def find_model_by_name(model_name: str) -> LLMModel | None:
    """Find a model by its name across all available models."""
    all_models = AVAILABLE_MODELS + OLLAMA_MODELS
    return next((model for model in all_models if model.model_name == model_name), None)


def get_models_list():
    """Get the list of models for API responses."""
    return [
        {
            "display_name": model.display_name,
            "model_name": model.model_name,
            "provider": model.provider.value
        }
        for model in AVAILABLE_MODELS
    ]


def _normalize_openai_base_url(base_url: str) -> str:
    """ChatOpenAI 会追加 /chat/completions，故 base_url 只能到 /v1。若已含 /v1/chat/completions 则先去掉该后缀。"""
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1/chat/completions"):
        return base_url[: -len("/chat/completions")]
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _get_api_key(api_keys: dict[str, str] | None, env_var: str, provider_value: str) -> str | None:
    """从环境变量或 api_keys 取 key。优先使用 .env（与 notebook 一致），其次请求/数据库中的 key。"""
    d = api_keys or {}
    from_env = os.getenv(env_var)
    from_dict = d.get(env_var) or d.get(provider_value)
    key = from_env or from_dict
    if env_var == "DASHSCOPE_API_KEY" and key:
        _src = "env" if from_env else "api_keys"
        _suffix = key[-6:] if len(key) > 10 else "?"
        logger.info("DASHSCOPE key source=%s suffix=...%s", _src, _suffix)
    return key


def get_model(
    model_name: str,
    model_provider: ModelProvider | str,
    api_keys: dict[str, str] | None = None,
) -> BaseChatModel | None:
    if isinstance(model_provider, str):
        model_provider = ModelProvider(model_provider)
    provider_value = model_provider.value

    api_keys_keys = list((api_keys or {}).keys())
    logger.info(
        "LLM get_model model=%s provider=%s api_keys_providers=%s",
        model_name, provider_value, api_keys_keys,
    )

    if model_provider == ModelProvider.GROQ:
        api_key = _get_api_key(api_keys, "GROQ_API_KEY", provider_value)
        if not api_key:
            # Print error to console
            print(f"API Key Error: Please make sure GROQ_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("Groq API key not found.  Please make sure GROQ_API_KEY is set in your .env file or provided via API keys.")
        return ChatGroq(model=model_name, api_key=SecretStr(api_key))
    elif model_provider == ModelProvider.OPENAI:
        # Get and validate API key
        api_key = _get_api_key(api_keys, "OPENAI_API_KEY", provider_value)
        base_url = os.getenv("OPENAI_API_BASE")
        if not api_key:
            # Print error to console
            print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("OpenAI API key not found.  Please make sure OPENAI_API_KEY is set in your .env file or provided via API keys.")
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url=base_url,
        )
    elif model_provider == ModelProvider.OPENAI_COMPATIBLE:
        api_key = _get_api_key(api_keys, "OPENAI_COMPATIBLE_API_KEY", provider_value)
        base_url = (api_keys or {}).get("OPENAI_COMPATIBLE_BASE_URL") or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        if not api_key:
            raise ValueError(
                "OpenAI 兼容 API key 未配置。请在 .env 中设置 OPENAI_COMPATIBLE_API_KEY，或通过 API keys 传入。"
            )
        if not base_url or not base_url.strip():
            raise ValueError(
                "OpenAI 兼容 Base URL 未配置。请在 .env 中设置 OPENAI_COMPATIBLE_BASE_URL（例如自建服务或第三方兼容 endpoint）。"
            )
        base_url = _normalize_openai_base_url(base_url)
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url=base_url,
        )
    elif model_provider == ModelProvider.IDEALAB:
        api_key = _get_api_key(api_keys, "IDEALAB_API_KEY", provider_value) or _get_api_key(
            api_keys, "OPENAI_COMPATIBLE_API_KEY", provider_value
        )
        base_url = (
            (api_keys or {}).get("IDEALAB_BASE_URL")
            or os.getenv("IDEALAB_BASE_URL")
            or (api_keys or {}).get("OPENAI_COMPATIBLE_BASE_URL")
            or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        )
        if not api_key:
            raise ValueError(
                "IDEALAB API key 未配置。请在 .env 中设置 IDEALAB_API_KEY 或 OPENAI_COMPATIBLE_API_KEY。"
            )
        if not base_url or not base_url.strip():
            raise ValueError(
                "IDEALAB Base URL 未配置。请在 .env 中设置 IDEALAB_BASE_URL 或 OPENAI_COMPATIBLE_BASE_URL。"
            )
        base_url = _normalize_openai_base_url(base_url)
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url=base_url,
        )
    elif model_provider == ModelProvider.DASHSCOPE:
        api_key = _get_api_key(api_keys, "DASHSCOPE_API_KEY", provider_value)
        base_url = (api_keys or {}).get("DASHSCOPE_BASE_URL") or os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        logger.info("DASHSCOPE base_url=%s", base_url)
        if not api_key:
            raise ValueError(
                "百炼 API key 未配置。请在 .env 中设置 DASHSCOPE_API_KEY（从阿里云百炼控制台获取）。"
            )
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url=base_url,
        )
    elif model_provider == ModelProvider.ANTHROPIC:
        api_key = _get_api_key(api_keys, "ANTHROPIC_API_KEY", provider_value)
        if not api_key:
            print(f"API Key Error: Please make sure ANTHROPIC_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("Anthropic API key not found.  Please make sure ANTHROPIC_API_KEY is set in your .env file or provided via API keys.")
        return ChatAnthropic(
            model_name=model_name,
            api_key=SecretStr(api_key),
        )
    elif model_provider == ModelProvider.DEEPSEEK:
        api_key = _get_api_key(api_keys, "DEEPSEEK_API_KEY", provider_value)
        if not api_key:
            print(f"API Key Error: Please make sure DEEPSEEK_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("DeepSeek API key not found.  Please make sure DEEPSEEK_API_KEY is set in your .env file or provided via API keys.")
        return ChatDeepSeek(model=model_name, api_key=SecretStr(api_key))
    elif model_provider == ModelProvider.GOOGLE:
        api_key = _get_api_key(api_keys, "GOOGLE_API_KEY", provider_value)
        if not api_key:
            print(f"API Key Error: Please make sure GOOGLE_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("Google API key not found.  Please make sure GOOGLE_API_KEY is set in your .env file or provided via API keys.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=SecretStr(api_key),
        )
    elif model_provider == ModelProvider.OLLAMA:
        # For Ollama, we use a base URL instead of an API key
        # Check if OLLAMA_HOST is set (for Docker on macOS)
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
        )
    elif model_provider == ModelProvider.OPENROUTER:
        api_key = _get_api_key(api_keys, "OPENROUTER_API_KEY", provider_value)
        if not api_key:
            print(f"API Key Error: Please make sure OPENROUTER_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("OpenRouter API key not found. Please make sure OPENROUTER_API_KEY is set in your .env file or provided via API keys.")
        
        # Get optional site URL and name for headers
        site_url = os.getenv("YOUR_SITE_URL", "https://github.com/virattt/ai-hedge-fund")
        site_name = os.getenv("YOUR_SITE_NAME", "AI Hedge Fund")
        
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url="https://openrouter.ai/api/v1",
            model_kwargs={
                "extra_headers": {
                    "HTTP-Referer": site_url,
                    "X-Title": site_name,
                },
            },
        )
    elif model_provider == ModelProvider.XAI:
        api_key = _get_api_key(api_keys, "XAI_API_KEY", provider_value)
        if not api_key:
            print(f"API Key Error: Please make sure XAI_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("xAI API key not found. Please make sure XAI_API_KEY is set in your .env file or provided via API keys.")
        return ChatXAI(model=model_name, api_key=SecretStr(api_key))
    elif model_provider == ModelProvider.GIGACHAT:
        if os.getenv("GIGACHAT_USER") or os.getenv("GIGACHAT_PASSWORD"):
            return GigaChat(model=model_name)
        else: 
            api_key = _get_api_key(api_keys, "GIGACHAT_API_KEY", provider_value) or os.getenv("GIGACHAT_CREDENTIALS")
            if not api_key:
                print("API Key Error: Please make sure api_keys is set in your .env file or provided via API keys.")
                raise ValueError("GigaChat API key not found. Please make sure GIGACHAT_API_KEY is set in your .env file or provided via API keys.")

            return GigaChat(credentials=api_key, model=model_name)
    elif model_provider == ModelProvider.AZURE_OPENAI:
        # Get and validate API key
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            # Print error to console
            print(f"API Key Error: Please make sure AZURE_OPENAI_API_KEY is set in your .env file.")
            raise ValueError("Azure OpenAI API key not found.  Please make sure AZURE_OPENAI_API_KEY is set in your .env file.")
        # Get and validate Azure Endpoint
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            # Print error to console
            print(f"Azure Endpoint Error: Please make sure AZURE_OPENAI_ENDPOINT is set in your .env file.")
            raise ValueError("Azure OpenAI endpoint not found.  Please make sure AZURE_OPENAI_ENDPOINT is set in your .env file.")
        # get and validate deployment name
        azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not azure_deployment_name:
            # Print error to console
            print(f"Azure Deployment Name Error: Please make sure AZURE_OPENAI_DEPLOYMENT_NAME is set in your .env file.")
            raise ValueError("Azure OpenAI deployment name not found.  Please make sure AZURE_OPENAI_DEPLOYMENT_NAME is set in your .env file.")
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment_name,
            api_key=SecretStr(api_key),
            api_version="2024-10-21",
        )
