# from repo root
cd grader_backend

# 1) Create env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure env vars (.env)
cat > .env << 'EOF'
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_API_KEY=YOUR_REAL_KEY_HERE
NIM_CHAT_MODEL=qwen/qwen3-next-80b-a3b-instruct
NIM_EMBED_MODEL=
EOF

# 4) Run backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
