"""기동 시 가중치를 볼륨에 맞추고 엔진을 올린 뒤 1초짜리 무음 참조로 한 번 생성해 커널을 데운다."""
import logging, os, tempfile, time
import numpy as np, soundfile as sf
logger = logging.getLogger("auk")


def preload():
    from app.handler import get_engine, _action_tts
    t = time.monotonic()
    get_engine()
    loaded = time.monotonic()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
    sf.write(tmp.name, (np.random.default_rng(0).standard_normal(24000) * 1e-3).astype(np.float32), 24000)
    try:
        _action_tts({"text": "준비 완료", "ref_audio": tmp.name, "gen_seconds": 1.0, "nfe": 8})
        note = f"warm {time.monotonic() - loaded:.1f}s"
    except Exception as exc:
        note = f"warm failed: {exc}"
    finally:
        os.unlink(tmp.name)
    logger.info(f"preload done in {time.monotonic() - t:.1f}s ({note})")
