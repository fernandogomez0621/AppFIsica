#!/bin/sh
# ============================================================================
# Entrypoint — prepara .streamlit/secrets.toml y verifica las API keys
# ============================================================================
# Las keys llegan como variables de entorno (vía env_file: .env en compose).
# La app las lee directamente de os.environ, soportando:
#   - GROQ_API_KEYS = "k1,k2,k3"   (lista por comas)
#   - GROQ_API_KEY / GROQ_API_KEY_BACKUP / GROQ_API_KEY_2 ... (con nombre)
# Aquí solo escribimos la config [rag]/[embeddings] y, por compatibilidad,
# la(s) key(s) con nombre en secrets.toml.
# ============================================================================
set -e

mkdir -p /app/.streamlit

# --- Config del modelo (siempre) -------------------------------------------
{
  [ -n "${GROQ_API_KEY:-}" ]         && echo "GROQ_API_KEY = \"${GROQ_API_KEY}\""
  [ -n "${GROQ_API_KEY_BACKUP:-}" ]  && echo "GROQ_API_KEY_BACKUP = \"${GROQ_API_KEY_BACKUP}\""
  echo ""
  echo "[rag]"
  echo "model_name = \"openai/gpt-oss-120b\""
  echo "temperature = 0.3"
  echo "max_tokens = 2048"
  echo ""
  echo "[embeddings]"
  echo "model_name = \"sentence-transformers/all-MiniLM-L6-v2\""
  echo "device = \"cpu\""
  echo "normalize_embeddings = true"
} > /app/.streamlit/secrets.toml

# --- Contar keys detectadas (cualquier forma) ------------------------------
N=0
[ -n "${GROQ_API_KEYS:-}" ] && N=$(printf '%s' "$GROQ_API_KEYS" | tr ',' '\n' | grep -c . || true)
[ -n "${GROQ_API_KEY:-}" ]        && N=$((N+1))
[ -n "${GROQ_API_KEY_BACKUP:-}" ] && N=$((N+1))
i=2
while [ $i -le 20 ]; do
  eval v=\${GROQ_API_KEY_$i:-}
  [ -n "$v" ] && N=$((N+1))
  i=$((i+1))
done

if [ "$N" -gt 0 ]; then
  echo "✅ $N API key(s) detectada(s) — modelo: openai/gpt-oss-120b — failover activo si N>1"
else
  echo "⚠️  No se detectó ninguna GROQ key. Copia .env.example a .env y añade al menos una."
  echo "    Genera claves en https://console.groq.com/keys"
fi

exec "$@"
