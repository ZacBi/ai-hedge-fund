# Prompt Management

Analyst and portfolio-manager prompts are centralized and can be managed via Langfuse.

## Where prompts live

- **Default content**: `src/prompts/registry.py` — one default per prompt (names like `hedge-fund/ben_graham`, `hedge-fund/portfolio_manager`).
- **Loading**: `src/prompts/loader.py` — at runtime, `get_prompt_template(name)` tries Langfuse first (when configured), then falls back to the registry.

## Using Langfuse

1. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` (same keys as for tracing).
2. Sync local defaults to Langfuse so prompts exist and are editable in the UI:
   ```bash
   uv run scripts/sync_prompts_to_langfuse.py
   ```
3. In the Langfuse UI you can edit prompts, create versions, and assign labels (e.g. `production`). Runtime uses the prompt version selected by label when Langfuse is configured.
4. In Langfuse, variables use double braces, e.g. `{{ticker}}`, `{{analysis_data}}`; the loader converts to LangChain’s format.

## Without Langfuse

If Langfuse is not configured or a prompt cannot be fetched, the app uses the default from `src/prompts/registry.py` with no change in behavior.
