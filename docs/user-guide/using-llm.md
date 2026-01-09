# Using the LLM

## Purpose
- Generate execution plans
- Summarize results and insights
- Assist with configuration and troubleshooting

## Configure providers
Set API keys via environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OLLAMA_HOST`

## Commands
- `proxima llm plan <circuit>`: propose execution plan
- `proxima llm summarize <report.json>`: summarize results

## Safety
- Consent checks gate LLM requests when enabled
- Avoid sending sensitive payloads unless required
