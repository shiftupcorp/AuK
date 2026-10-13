"""S3 모델 레지스트리 → 볼륨 지연 동기화(boto3). 디렉터리 안 파일이 하나라도 비면 prefix 전체를 받는다."""
import logging, os
logger = logging.getLogger("auk")


def _split(uri):
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {uri}")
    bucket, key = (uri[5:].split("/", 1) + [""])[:2]
    return bucket, key


def sync_prefix(s3_prefix, local_dir, required=()):
    """s3_prefix 아래 전체를 local_dir로 받는다(이미 있는 파일은 크기가 같으면 건너뜀). required 파일이 다 있으면 S3를 보지 않는다."""
    if required and all(os.path.exists(os.path.join(local_dir, r)) for r in required):
        return local_dir
    from concurrent.futures import ThreadPoolExecutor
    bucket, prefix = _split(s3_prefix.rstrip("/") + "/")
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    todo = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            dst = os.path.join(local_dir, rel)
            if os.path.exists(dst) and os.path.getsize(dst) == obj["Size"]:
                continue
            todo.append((obj["Key"], dst))

    def fetch(key_dst):
        key, dst = key_dst
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        client.download_file(bucket, key, tmp, Config=_transfer_config())
        os.replace(tmp, dst)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, todo))
    n = len(todo)
    logger.info(f"synced {n} files from {s3_prefix} -> {local_dir}")
    missing = [r for r in required if not os.path.exists(os.path.join(local_dir, r))]
    if missing:
        raise FileNotFoundError(f"{s3_prefix} lacks {missing}")
    return local_dir


def _client():
    """S3_ACCELERATE=1이면 가속 엔드포인트(EU pod ↔ ap-northeast-2 버킷에서 6 MB/s → 200 MB/s 실측, 2026-10-13)."""
    import boto3
    from botocore.config import Config
    accel = os.environ.get("S3_ACCELERATE", "0") == "1"
    return boto3.client("s3", config=Config(s3={"use_accelerate_endpoint": accel}, max_pool_connections=64))


def _transfer_config():
    from boto3.s3.transfer import TransferConfig
    return TransferConfig(max_concurrency=16, multipart_chunksize=64 * 1024 * 1024)


def download_object(uri, dst):
    bucket, key = _split(uri)
    _client().download_file(bucket, key, dst, Config=_transfer_config())
    return dst
