"""AuK(한국어 FT v5) pod/serverless 핸들러.

event["input"] = {"action": "tts", "text": "...", "ref_audio": "s3://… | https://… | /path | (ref_audio_base64)",
                  "ref_text": "참조 대사(선택, 있으면 공식 길이 추정)", "char": "evie", "instruction": "(선택) 영어 지시문",
                  "gen_seconds": (선택) 길이 강제, "speed": 1.0, "seed": 666, "nfe": 32, "cfg": 2.0}
→ {"model": "auk-ko-v5", "char", "sample_rate": 24000, "audio_base64": wav, "gen_seconds", "length_how", "prompt"}

참조 오디오는 메시지의 audio 항목(Qwen 조건)과 audio= 인자(DiT 접두) 양쪽에 넣는다 — 공식 형식.
# ref: voice/docs/49_movie_mining.md §7.5
"""
import base64, io, logging, math, os, re, tempfile, threading, time
import numpy as np
import soundfile as sf
import torch, torchaudio
from app.s3util import download_object, sync_prefix

logger = logging.getLogger("auk")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MODEL_NAME = os.environ.get("AUK_MODEL_NAME", "auk-ko-v5")
VOLUME_ROOT = os.environ.get("VOICE_VOLUME_ROOT", "/workspace")
MODEL_DIR = os.environ.get("AUK_MODEL_DIR", os.path.join(VOLUME_ROOT, "auk", "v5_13554"))
QWEN_DIR = os.environ.get("AUK_QWEN_DIR", os.path.join(VOLUME_ROOT, "auk", "qwen2.5-omni-3b"))
S3_PREFIX = os.environ.get("AUK_S3_PREFIX", "s3://shiftup-enterprise-ai-service/voice/model_registry/auk")
CONFIG = os.environ.get("AUK_CONFIG", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "pod.yaml"))
DTYPE = os.environ.get("AUK_DTYPE", "bf16")
MAX_REF_SECONDS = float(os.environ.get("AUK_MAX_REF_SECONDS", "20"))
MAX_GEN_SECONDS = float(os.environ.get("AUK_MAX_GEN_SECONDS", "30"))

ZS = 'Say the following with the same voice: "{text}"'
ZS_DESC = 'Generate speech with the same voice based on the following description: "{desc}". The content to speak is: "{text}".'
SPAN_TAGS = {"whisper", "shout", "laughing", "crying"}
TAG_RE = re.compile(r"\[([a-z -]+)\]")

_engine = None
_lock = threading.Lock()


def ensure_weights():
    sync_prefix(f"{S3_PREFIX}/v5_13554", MODEL_DIR, required=("model.safetensors", "vae.safetensors"))
    sync_prefix(f"{S3_PREFIX}/qwen2.5-omni-3b", QWEN_DIR, required=("config.json", "model.safetensors.index.json", "model-00001-of-00003.safetensors"))


def get_engine():
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                from auk.infer.infer_auk import AukInfer
                ensure_weights()
                t = time.monotonic()
                _engine = AukInfer(CONFIG, os.path.join(MODEL_DIR, "model.safetensors"), dtype=DTYPE, device="cuda", qwen_path=QWEN_DIR)
                logger.info(f"engine ready in {time.monotonic() - t:.1f}s (model={MODEL_DIR}, qwen={QWEN_DIR})")
    return _engine


def _resolve_ref(inp):
    """ref_audio_base64 > ref_audio(s3/https/local). 임시 파일 경로를 돌려준다(원격·base64는 지움 대상)."""
    b64 = inp.get("ref_audio_base64")
    if b64:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.write(base64.b64decode(b64)); tmp.close()
        return tmp.name, True
    path = inp.get("ref_audio") or inp.get("ref_audio_path") or inp.get("prompt_wav")
    if not path:
        raise ValueError("ref_audio required (s3://, https://, local path, or ref_audio_base64)")
    if path.startswith("s3://") or path.startswith("https://"):
        if path.startswith("https://"):
            host, key = path[len("https://"):].split("/", 1)
            bucket = host.split(".s3.")[0] if ".s3." in host else None
            if bucket is None:
                raise ValueError(f"cannot parse S3 URL: {path}")
            path = f"s3://{bucket}/{key}"
        ext = os.path.splitext(path)[-1] or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False); tmp.close()
        download_object(path, tmp.name)
        return tmp.name, True
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path, False


def _prepare_ref(path):
    """모노·24k wav로 정규화하고 MAX_REF_SECONDS를 넘으면 앞부분만 쓴다(길이 상한만, 음색 불변)."""
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000); sr = 24000
    secs = audio.shape[-1] / sr
    if secs > MAX_REF_SECONDS:
        logger.info(f"ref {secs:.1f}s > {MAX_REF_SECONDS}s, using the first {MAX_REF_SECONDS}s")
        audio = audio[:, : int(MAX_REF_SECONDS * sr)]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
    torchaudio.save(tmp.name, audio, sr)
    return tmp.name, audio.shape[-1] / sr


def estimate_seconds(ref_path, ref_text, text, speed):
    """참조 대사가 있으면 공식 추정(참조 길이 × 바이트 비), 없으면 글자 수 추정. 태그는 점 0.8초·구간 1.0초 가산."""
    from auk.infer.infer_auk import get_gen_duration
    plain = re.sub(r"\s*\[[a-z -]+\]\s*", " ", text).strip()
    tags = TAG_RE.findall(text)
    extra = sum(1.0 if t in SPAN_TAGS else 0.8 for t in tags)
    if ref_text and ref_text.strip():
        base = get_gen_duration(audio=ref_path, ref_text=ref_text.strip(), gen_text=plain, speed=speed); how = "official(ref_text)"
    else:
        base = (0.19 * len(re.sub(r"\s", "", plain)) + 0.5) / speed; how = "chars"
    return round(max(0.5, min(base + extra, MAX_GEN_SECONDS)), 2), how


def _wav_b64(audio, sr):
    buf = io.BytesIO(); sf.write(buf, np.asarray(audio.squeeze(0).numpy(), dtype=np.float32), sr, format="WAV"); buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _action_tts(inp):
    text = (inp.get("text") or "").strip()
    if not text:
        raise ValueError("text required")
    speed = float(inp.get("speed") or 1.0); seed = int(inp.get("seed") or 666)
    nfe = int(inp.get("nfe") or 32); cfg = float(inp.get("cfg") or 2.0)
    instruction = (inp.get("instruction") or "").replace('"', "'").strip().rstrip(".")
    src, is_tmp = _resolve_ref(inp)
    try:
        ref_path, ref_secs = _prepare_ref(src)
    finally:
        if is_tmp:
            os.unlink(src)
    try:
        ref_text = inp.get("ref_text") or inp.get("prompt_text") or ""
        if inp.get("gen_seconds"):
            sec, how = round(max(0.5, min(float(inp["gen_seconds"]), MAX_GEN_SECONDS)), 2), "explicit"
        else:
            sec, how = estimate_seconds(ref_path, ref_text, text, speed)
        prompt = ZS_DESC.format(desc=instruction, text=text) if instruction else ZS.format(text=text)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "audio", "audio_url": ref_path}]}]
        engine = get_engine()
        t = time.monotonic()
        with _lock:
            audio, sr = engine.generate(messages, audio=ref_path, gen_seconds=sec, nfe=nfe, cfg_strength=cfg, sway_sampling_coef=-1.0, seed=seed)
        elapsed = time.monotonic() - t
    finally:
        os.unlink(ref_path)
    audio = audio.detach().cpu().float()
    logger.info(f"tts {sec}s ({how}) ref {ref_secs:.1f}s seed {seed} in {elapsed:.1f}s")
    return {"model": MODEL_NAME, "char": inp.get("char"), "sample_rate": sr, "audio_base64": _wav_b64(audio, sr),
            "gen_seconds": sec, "length_how": how, "ref_seconds": round(ref_secs, 2), "prompt": prompt, "seed": seed, "elapsed": round(elapsed, 2)}


_DISPATCH = {"tts": _action_tts, "health": lambda inp: {"status": "ok", "model": MODEL_NAME, "loaded": _engine is not None}}


def handler(event):
    inp = event.get("input") or {}
    action = inp.get("action") or "tts"
    fn = _DISPATCH.get(action)
    if fn is None:
        return {"error": f"unknown action: {action}", "actions": sorted(_DISPATCH)}
    return fn(inp)
