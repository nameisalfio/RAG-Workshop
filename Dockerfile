# Image of the Python services. The API (backend) and the ingestion job share it.
FROM condaforge/miniforge3:26.7.2-0

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Libraries that the PDF and OCR stack loads at run time.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN conda create --name rag-workshop --yes python=3.12 pip \
    && conda clean --all --yes
ENV PATH=/opt/conda/envs/rag-workshop/bin:$PATH

WORKDIR /app

# CPU build of PyTorch: the default Linux wheels bundle GPU libraries that the demo never uses.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dependencies are installed before the source is copied, so that editing the code does not reinstall them.
COPY pyproject.toml ./
RUN mkdir src && touch src/__init__.py && pip install --editable .
COPY src ./src

EXPOSE 8000
CMD ["uvicorn", "src.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
