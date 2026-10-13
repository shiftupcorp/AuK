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
    import boto3
    bucket, prefix = _split(s3_prefix.rstrip("/") + "/")
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            dst = os.path.join(local_dir, rel)
            if os.path.exists(dst) and os.path.getsize(dst) == obj["Size"]:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".part"
            client.download_file(bucket, obj["Key"], tmp)
            os.replace(tmp, dst)
            n += 1
    logger.info(f"synced {n} files from {s3_prefix} -> {local_dir}")
    missing = [r for r in required if not os.path.exists(os.path.join(local_dir, r))]
    if missing:
        raise FileNotFoundError(f"{s3_prefix} lacks {missing}")
    return local_dir


def download_object(uri, dst):
    import boto3
    bucket, key = _split(uri)
    boto3.client("s3").download_file(bucket, key, dst)
    return dst
