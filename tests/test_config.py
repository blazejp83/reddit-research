import reddit_research.core.config as config
from reddit_research.core.config import load_config


def test_dotenv_populates_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("BRAVE_SEARCH_API_KEY=from-dotenv\nREDDIT_CLIENT_ID=id-dotenv\n")
    monkeypatch.setattr(config, "DOTENV_SEARCH_PATHS", [env_file])
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)

    cfg = load_config()
    assert cfg.brave_api_key == "from-dotenv"
    assert cfg.reddit.client_id == "id-dotenv"


def test_real_env_wins_over_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("BRAVE_SEARCH_API_KEY=from-dotenv\n")
    monkeypatch.setattr(config, "DOTENV_SEARCH_PATHS", [env_file])
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "from-real-env")

    cfg = load_config()
    assert cfg.brave_api_key == "from-real-env"


def test_secrets_not_read_from_toml(tmp_path, monkeypatch):
    toml = tmp_path / "reddit-research.toml"
    toml.write_text('[reddit]\nclient_secret = "leaked"\nbackend = "arctic"\n')
    monkeypatch.setattr(config, "DOTENV_SEARCH_PATHS", [])
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)

    cfg = load_config(toml)
    assert cfg.reddit.client_secret is None
    assert cfg.reddit.backend == "arctic"
