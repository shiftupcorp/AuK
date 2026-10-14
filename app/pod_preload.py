"""기동 시 가중치를 볼륨에 맞추고 엔진을 올린 뒤 실제 요청 모양(7초 참조·4초/2.5초 생성·NFE 32)으로 두 번 생성해 데운다. 1초·NFE 8 워밍업으로는 첫 실요청이 29~89초 걸렸다(docs/50 §6)."""
import logging, os, tempfile, time
import numpy as np, soundfile as sf
logger = logging.getLogger("auk")


def preload():
    from app.handler import get_engine, _action_tts
    t = time.monotonic()
    get_engine()
    loaded = time.monotonic()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
    sf.write(tmp.name, (np.random.default_rng(0).standard_normal(7 * 24000) * 1e-3).astype(np.float32), 24000)
    try:
        for secs in (4.0, 2.5):
            _action_tts({"text": "준비 완료. 첫 요청이 느리지 않도록 실제 길이로 한 번 돈다.", "ref_audio": tmp.name, "gen_seconds": secs})
        note = f"warm {time.monotonic() - loaded:.1f}s"
    except Exception as exc:
        note = f"warm failed: {exc}"
    finally:
        os.unlink(tmp.name)
    logger.info(f"preload done in {time.monotonic() - t:.1f}s ({note})")
