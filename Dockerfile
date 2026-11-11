FROM python:3.11-slim-bookworm
WORKDIR /opt/product
COPY . .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
CMD ["sh", "-c", "mkdir -p /tmp/product-run && cp -R /opt/product/. /tmp/product-run/ && cd /tmp/product-run && python3 -m unittest discover -s tests -v && python3 -m compileall -q trialvault && python3 -m trialvault demo"]
