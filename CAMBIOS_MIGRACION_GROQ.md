# Migración de modelo Groq (junio 2026)

## Motivo
Groq deprecó `llama-3.3-70b-versatile` el **17/06/2026**. Las llamadas al LLM
dejaron de funcionar. Reemplazo recomendado por Groq: `openai/gpt-oss-120b`.

## Archivos modificados
- `src/educational_rag.py` — modelo ahora configurable vía `secrets.toml`
  (`[rag] model_name`), por defecto `openai/gpt-oss-120b`. Se cambió
  `max_tokens` → `max_completion_tokens=4096` y se añadió `reasoning_effort="medium"`
  (GPT-OSS es modelo de razonamiento).
- `rag_app.py` — `ChatGroq(... model_name="openai/gpt-oss-120b")`.
- `test_rag_simple.py` — modelo actualizado en las dos llamadas.
- `.streamlit/secrets.toml` — `model_name = "openai/gpt-oss-120b"`.

## Acciones pendientes (hazlas tú)
1. **Rotar las claves de Groq**: las anteriores estaban commiteadas en el repo y
   quedan comprometidas. Genera una nueva en https://console.groq.com/keys y
   reemplaza `TU_NUEVA_API_KEY_DE_GROQ_AQUI` en `.streamlit/secrets.toml`.
2. Las claves viejas siguen en el **historial de Git**. Límpialo con
   `git filter-repo` o BFG si el repo es público.
3. Se añadió un `.gitignore` que ya excluye `secrets.toml`, `__pycache__/` y
   los entornos virtuales para que no se vuelvan a subir.

## Probar
```bash
pip install -r requirements.txt
python test_rag_simple.py     # prueba rápida de la API
streamlit run app.py          # app completa
```
