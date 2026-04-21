from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM - OpenAI
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

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    upload_dir: str = "/app/data/uploads"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
