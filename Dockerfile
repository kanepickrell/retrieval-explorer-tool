# Retrieval Explorer: the React UI and the FastAPI backend in one container, on one port.
#
#   docker build -t oui-agents/retrieval-explorer:dev .
#   docker run -p 127.0.0.1:5180:8000 -v $PWD/server/datasets:/app/server/datasets oui-agents/retrieval-explorer:dev
#
# Datasets (indexes, exports) are NOT baked in: mount server/datasets.

FROM node:22-alpine3.20 AS ui
WORKDIR /ui
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY index.html postcss.config.js tailwind.config.js tsconfig.json vite.config.ts ./
COPY public ./public
COPY src ./src
# Same origin as the backend, so API calls are relative.
ENV VITE_API_URL=""
RUN npm run build

FROM python:3.12-slim
WORKDIR /app/server
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    STATIC_DIR=/app/static TIKTOKEN_CACHE_DIR=/app/tiktoken
COPY server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && (python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base')" || true)
COPY server/ ./
COPY --from=ui /ui/dist /app/static
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
