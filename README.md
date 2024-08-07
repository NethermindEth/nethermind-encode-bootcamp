# Streamlit and Langfuse showcase

## Running the application

1. Run required Langfuse and Postgres:

```bash
docker-compose up
```

2. Go to `http://localhost:3000/` and create a new Project there
3. In the Project settings create a new API keys (remember both public and privte key)
4. Set up environment variables with the key from the previous point (Powershell example):

```powershell
$Env:LANGFUSE_SECRET_KEY="sk-lf-..."
$Env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
$Env:LANGFUSE_HOST="http://localhost:3000"
```

5. Set up also `OPENAI_API_KEY` environment variable
6. Run the main script:

```bash
streamlit run .\src\chat.py
```

7. You can also run the evaluation script:

```bash
python .\src\evaluate.py
```

*Note: This repository was created as a part of [Encode AI Bootcamp](https://www.encode.club/ai-bootcamp).*