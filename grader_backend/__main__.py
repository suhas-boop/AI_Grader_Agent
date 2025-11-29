# grader_backend/__main__.py

import os
import uvicorn
from .main import app


def main():
    host = os.getenv("GRADER_BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("GRADER_BACKEND_PORT", "8000"))
    reload = os.getenv("GRADER_BACKEND_RELOAD", "true").lower() == "true"

    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
