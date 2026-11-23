FROM python:3.12-slim AS a-eye

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user for runtime + gosu for privilege drop
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r aeye && useradd -r -g aeye -d /app aeye \
    && mkdir -p /app/data/thumbnails \
    && chown -R aeye:aeye /app/data \
    && chmod +x /app/entrypoint.sh

# display the build date in the nav bar, for tracking whether I'm running the right dev build
RUN sed -i "s#{{ build_date }}#<div class=\"nav-status\"><span style=\"font-size:smaller;\">Build $(date '+%Y-%m-%d_%H:%M')</span></div>#" frontend/templates/base.html

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
