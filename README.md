# Quick install

1. Download the [github mcp server zip](https://github.com/github/github-mcp-server/releases) and copy the into the root of the repo
2. Create a `.env` file to store anthropic and github tokens
3. Install requirements in `requirements.txt`
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app in watch mode
   ```bash
   uvicorn app:app --reload
   ```