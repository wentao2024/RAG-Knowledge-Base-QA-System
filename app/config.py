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

    # Chunking — parent chunk (large chunk fed to the LLM)
    chunk_size: int = 512
    chunk_overlap: int = 64
    # Chunking — child chunk (small chunk for embedding, parent-child mode only)
    child_chunk_size: int = 150
    child_chunk_overlap: int = 30
    enable_parent_child: bool = True

    # Redis & Memory
    redis_url: str = "redis://localhost:6379"
    session_ttl: int = 604800           # 7 days
    memory_ttl: int = 2592000           # 30 days
    memory_fact_extract_every: int = 3  # extract facts every N turns
    memory_compress_after: int = 15     # compress history into summary after N turns
    memory_top_k: int = 3              # number of memories injected per retrieval
    enable_user_memory: bool = True

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    upload_dir: str = "/app/data/uploads"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # extra fields in .env are silently ignored
    )


settings = Settings()
