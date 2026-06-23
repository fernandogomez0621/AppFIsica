# 🐳 Despliegue con Docker

Forma más rápida y reproducible de levantar el proyecto. No necesitas instalar
Python ni las dependencias a mano: todo queda dentro de los contenedores.

Levanta **dos servicios**:

| Servicio | Qué es | URL |
|----------|--------|-----|
| `app`    | La aplicación Streamlit | http://localhost:8501 |
| `docs`   | La documentación Sphinx (sitio estático) | http://localhost:8502 |

## Requisitos
- Docker y Docker Compose ([Docker Desktop](https://www.docker.com/products/docker-desktop/) en Windows/Mac, o `docker` + plugin compose en Linux).

## Pasos

```bash
# 1. Configura tu clave de Groq
cp .env.example .env
#    edita .env y pon tu GROQ_API_KEY (https://console.groq.com/keys)

# 2. Construye y levanta ambos servicios
docker compose up --build

# 3. Abre:
#    App  → http://localhost:8501
#    Docs → http://localhost:8502
```

Detener: `Ctrl+C`. En segundo plano: `docker compose up -d --build` y luego `docker compose down`.

Levantar solo uno: `docker compose up app`  ó  `docker compose up docs`.

## Qué hace el setup
- **app**: imagen `python:3.12-slim` con `torch` en versión CPU (evita ~2.5 GB
  de wheels CUDA). El entrypoint genera `.streamlit/secrets.toml` desde
  `GROQ_API_KEY`, con el modelo fijado en `openai/gpt-oss-120b`. Volúmenes para
  que `data/`, `models/` y `chroma_rbm_db/` **persistan** entre reinicios.
- **docs**: reutiliza la misma imagen. Al arrancar ejecuta `sphinx-build` y sirve
  el HTML con el servidor estático de Python. Usa `PYTHONPATH=/app` para que
  autodoc pueda importar `app` y `src.*`.

## Cambiar los puertos
Edita la sección `ports` en `docker-compose.yml`. El formato es `HOST:CONTENEDOR`.
Ej.: para servir la app en el 1111 y la doc en el 2222 (como el despliegue original):

```yaml
  app:
    ports: ["1111:8501"]
  docs:
    ports: ["2222:8000"]
```

## Notas
- La primera construcción tarda (descarga torch, transformers, etc.). Las
  siguientes usan caché y son mucho más rápidas.
- El build de la doc puede mostrar *avisos* de autodoc (p. ej. al importar
  módulos que usan Streamlit); son normales y no detienen la generación.
- Se quitó `tensorflow` (no se usaba) y la dependencia rota `fitz` del
  `requirements.txt`.
- Los contenedores corren como root (suficiente para uso local). Para producción:
  usuario no-root + reverse proxy (nginx/traefik) con HTTPS.
