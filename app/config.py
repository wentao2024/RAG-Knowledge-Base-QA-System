from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM - OpenAI compatible
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    embed_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # Vector Store
    chroma_persist_dir: str = "/app/data/chroma"
    collection_name: str = "rag_documents"

    # Retrieval
    top_k: int = 5
    rrf_k: int = 60
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    enable_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"

    # Chunking — parent chunk (喂给 LLM 的大块)
    chunk_size: int = 512
    chunk_overlap: int = 64
    # Chunking — child chunk (做 Embedding 的小块，parent-child 模式专用)
    child_chunk_size: int = 150
    child_chunk_overlap: int = 30
    enable_parent_child: bool = True

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    upload_dir: str = "/app/data/uploads"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # .env 里有多余字段不报错
    )


settings = Settings()
