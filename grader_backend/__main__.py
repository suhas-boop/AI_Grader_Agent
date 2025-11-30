# grader_backend/__main__.py

import os
import uvicorn
from grader_backend.main import app  # uses your existing FastAPI app


def main() -> None:
    
    host = os.getenv("GRADER_BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("GRADER_BACKEND_PORT", "8000"))
    reload_flag = os.getenv("GRADER_BACKEND_RELOAD", "true").lower() == "true"

    uvicorn.run(app, host=host, port=port, reload=reload_flag)


if __name__ == "__main__":
    main()
