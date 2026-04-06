"""OpenEnv-compatible server entry point."""

import uvicorn

from api import app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the existing FastAPI app through an OpenEnv-friendly entry point."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
