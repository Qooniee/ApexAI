# --- ステージ1: ビルド環境 (uv + PyTorch) ---
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# システムパッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# uvのインストール (公式推奨の方法)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 作業ディレクトリ
WORKDIR /workspace

# Python仮想環境をuvで作成
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv venv /opt/venv --python python3.12
ENV PATH="/opt/venv/bin:$PATH"

# PyTorchを先にインストール (CUDA 12.8対応)
RUN uv pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# プロジェクトファイル全体をコピー（README.mdを含む）
COPY . .

# プロジェクトの依存関係とプロジェクト自体をインストール
# UV_PROJECT_ENVIRONMENTが設定されているので/opt/venvにインストールされる
RUN uv sync --no-dev


# --- ステージ2: 実行環境 (最終的なイメージ) ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# CUDAライブラリへのパスを環境変数に明示的に設定
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 実行に必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        git \
        graphviz \
        htop \
        nvtop \
        tree \
        vim \
        curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ビルドステージから、uvコマンドと完成した仮想環境をコピー
COPY --from=builder /bin/uv /bin/uvx /bin/
COPY --from=builder /opt/venv /opt/venv

# 仮想環境を有効化
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 作業ディレクトリ
WORKDIR /workspace

# デフォルトのコマンドは設定せず、docker-compose.yamlで指定する
