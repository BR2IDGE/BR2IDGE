import contextlib
import inspect
import os
import types
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _exists(p: Path) -> bool:
    try:
        return p.exists() and (p.is_dir() or p.stat().st_size > 0)
    except Exception:
        return False

def _npz_save(obj: dict[str, Any], p: Path) -> None:
    np.savez_compressed(p, **{k: v for k, v in obj.items() if v is not None})

def _npz_load(p: Path) -> dict[str, np.ndarray]:
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}

def _normalize_key(model_name: str) -> str:
    key = (model_name or "").lower().strip()

    for suf in ("model", "wrapper"):
        if key.endswith(suf):
            key = key[: -len(suf)].strip()

    if "lightfm" in key:
        return "lightfm"
    if "als" in key:
        return "als"
    if "nmf" in key:
        return "nmf"
    if "libreco" in key or "deepfm" in key:
        return "deepfm_libreco"

    return key


def _maybe_cast_fp16_state(state: dict) -> dict:
    if os.getenv("CKPT_TORCH_FP16", "0") != "1":
        return state
    try:
        import torch
        out = {}
        for k, v in state.items():
            if hasattr(v, "dtype") and v.dtype in (torch.float32, torch.float64):
                out[k] = v.half()
            else:
                out[k] = v
        return out
    except Exception:
        return state

def _torch_save(wrapper, p: Path) -> None:
    import torch
    model = None
    optimizer = getattr(wrapper, "optimizer", None)
    if hasattr(wrapper, "state_dict"):
        model = wrapper
    elif hasattr(wrapper, "model") and hasattr(wrapper.model, "state_dict"):
        model = wrapper.model
    else:
        raise ValueError("Object does not have state_dict nor a .model with state_dict.")

    payload = {"model": _maybe_cast_fp16_state(model.state_dict())}
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        with np.errstate(all="ignore"):
            try:
                payload["optimizer"] = optimizer.state_dict()
            except Exception:
                pass
    torch.save(payload, p, _use_new_zipfile_serialization=True)

def _torch_load(wrapper, p: Path) -> bool:
    import torch
    state = torch.load(p, map_location="cpu")
    target = getattr(wrapper, "model", None) if hasattr(wrapper, "model") else wrapper
    if target is None or not hasattr(target, "load_state_dict"):
        print("[adapters.py] _torch_load failed: target has no load_state_dict().")
        return False
    target.load_state_dict(state.get("model", {}), strict=False)
    if hasattr(wrapper, "optimizer") and "optimizer" in state:
        with np.errstate(all="ignore"):
            try:
                wrapper.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
    return True


def _keras_save(model, p: Path) -> None:
    weights_path = str(p) + ".weights.h5"
    model.save_weights(weights_path)

def _keras_load(wrapper, p: Path) -> bool:
    weights_path = str(p) + ".weights.h5"
    if getattr(wrapper, "model", None) is not None:
        try:
            wrapper.model.load_weights(weights_path)
            return True
        except Exception:
            setattr(wrapper, "_pending_keras_weights", weights_path)
            return True
    else:
        setattr(wrapper, "_pending_keras_weights", weights_path)
        return True


def _lightfm_save(wrapper, p: Path) -> None:
    m = getattr(wrapper, "model", None)
    if m is None:
        print("[adapters.py] WARNING: _lightfm_save: wrapper.model is None.")
        return

    candidates = {
        "user_embeddings": ["user_embeddings"],
        "item_embeddings": ["item_embeddings"],
        "user_biases": ["user_biases", "user_biases_"],
        "item_biases": ["item_biases", "item_biases_"],
    }

    obj = {"no_components": np.int32(int(getattr(m, "no_components", 0)))}
    loaded = []
    for key, names in candidates.items():
        for name in names:
            val = getattr(m, name, None)
            if val is not None:
                obj[key] = np.ascontiguousarray(val).astype(np.float32, copy=False)
                loaded.append(key)
                break

    if loaded:
        print(f"[adapters.py] Saving LightFM arrays: {', '.join(sorted(set(loaded)))}")
        _npz_save(obj, p.with_suffix(".npz"))
    else:
        print("[adapters.py] WARNING: _lightfm_save: no weight attributes found.")


def _lightfm_load(wrapper, p: Path) -> bool:
    try:
        from lightfm import LightFM
    except ImportError:
        print("[adapters.py] Error: LightFM is not installed.")
        return False

    npz_path = p.with_suffix(".npz")
    if not _exists(npz_path):
        return False

    z = _npz_load(npz_path)

    model = getattr(wrapper, "model", None)
    if model is None:
        print("[adapters.py] _lightfm_load: wrapper.model was None. Creating fallback...")
        params = getattr(wrapper, "model_params", {}) or {}
        if "no_components" not in params and "embedding_dim" in params:
            params = dict(params)
            params["no_components"] = int(params["embedding_dim"])
        valid_keys = inspect.signature(LightFM.__init__).parameters.keys()
        init_params = {k: v for k, v in params.items() if k in valid_keys}

        if "no_components" in z:
            with contextlib.suppress(Exception):
                init_params["no_components"] = int(np.asarray(z["no_components"]).reshape(()))
        elif "no_components" not in init_params:
            init_params["no_components"] = 10

        print(f"[adapters.py] ...recreating LightFM with params: {init_params}")
        model = LightFM(**init_params)
        wrapper.model = model

    key_to_attr = {
        "user_embeddings": "user_embeddings",
        "item_embeddings": "item_embeddings",
        "user_biases": "user_biases",
        "item_biases": "item_biases",
    }

    print("[adapters.py] Hydrating LightFM with checkpoint weights...")
    loaded_count = 0
    for key, attr in key_to_attr.items():
        if key in z and z[key] is not None:
            setattr(model, attr, np.ascontiguousarray(z[key]).astype(np.float32, copy=False))
            loaded_count += 1

    if getattr(model, "user_biases", None) is None and getattr(model, "user_embeddings", None) is not None:
        model.user_biases = np.zeros(model.user_embeddings.shape[0], dtype=np.float32)
        loaded_count += 1
    if getattr(model, "item_biases", None) is None and getattr(model, "item_embeddings", None) is not None:
        model.item_biases = np.zeros(model.item_embeddings.shape[0], dtype=np.float32)
        loaded_count += 1

    if "no_components" in z:
        with contextlib.suppress(Exception):
            setattr(model, "no_components", int(np.asarray(z["no_components"]).reshape(())))

    if loaded_count == 0:
        print(f"[adapters.py] WARNING: no valid weights in {npz_path}")
        return False

    try:
        setattr(model, "fitted", True)
    except Exception:
        pass

    try:
        model.fitted = True
    except Exception:
        pass

    try:
        if hasattr(model, "_check_initialized"):
            def _ok(self):
                if getattr(self, "user_embeddings", None) is None or getattr(self, "item_embeddings", None) is None:
                    raise ValueError("LightFM checkpoint without embeddings; training is required.")

            model._check_initialized = types.MethodType(_ok, model)
    except Exception:
        pass

    print("[adapters.py] Model marked as 'fitted' and _check_initialized() neutralized.")
    return True


def _np_maybe_fp16(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    return arr.astype(np.float16) if os.getenv("CKPT_FP16", "0") == "1" else arr

def _als_save(wrapper, p: Path) -> None:
    obj: dict[str, Any] = {}
    for name in ("U", "V", "b_u", "b_i"):
        val = getattr(wrapper, name, None)
        if isinstance(val, np.ndarray) and val.dtype.kind == "f":
            obj[name] = _np_maybe_fp16(val)
        else:
            obj[name] = val
    obj["global_mean"] = np.asarray(float(getattr(wrapper, "global_mean", 0.0)))
    _npz_save(obj, p)

def _als_load(wrapper, p: Path) -> bool:
    z = _npz_load(p)
    if "U" in z and "V" in z:
        wrapper.U = np.asarray(z["U"], dtype=np.float32)
        wrapper.V = np.asarray(z["V"], dtype=np.float32)
        if "b_u" in z and z["b_u"] is not None:
            wrapper.b_u = np.asarray(z["b_u"], dtype=np.float32)
        if "b_i" in z and z["b_i"] is not None:
            wrapper.b_i = np.asarray(z["b_i"], dtype=np.float32)
        gm = z.get("global_mean", np.asarray(0.0))
        with contextlib.suppress(Exception):
            gm = float(np.asarray(gm).reshape(()))
        wrapper.global_mean = float(gm)
        if getattr(wrapper, "model", None) is None:
            wrapper.model = True
        return True
    return False

def _nmf_save(wrapper, p: Path) -> None:
    obj: dict[str, Any] = {}
    W = getattr(wrapper, "_W", None)
    H = getattr(wrapper, "_H", None)
    if W is not None:
        obj["W"] = _np_maybe_fp16(W)
    if H is not None:
        obj["H"] = _np_maybe_fp16(H)

    if "W" in obj and "H" in obj:
        print("[adapters.py] Saving 2 NMF arrays (W, H)...")
        _npz_save(obj, p)
    else:
        print("[adapters.py] WARNING: _nmf_save failed. _W or _H not found.")


def _nmf_load(wrapper, p: Path) -> bool:
    z = _npz_load(p)
    if "W" in z and "H" in z:
        wrapper._W = np.asarray(z["W"], dtype=np.float32)
        wrapper._H = np.asarray(z["H"], dtype=np.float32)
        if getattr(wrapper, "model", None) is None:
            wrapper.model = True
        print("[adapters.py] NMF factors (W, H) loaded.")
        return True
    print(f"[adapters.py] WARNING: _nmf_load failed at {p}")
    return False


def _libreco_save(wrapper, p: Path) -> None:
    m = getattr(wrapper, "model", None)
    if m is None:
        print("[adapters.py] WARNING: _libreco_save: wrapper.model is None.")
        return
    path_dir = str(p.parent)
    model_name = p.name
    print(f"[adapters.py] Saving LibReCo model in {path_dir} (name={model_name})...")
    m.save(path=path_dir, model_name=model_name, manual=True)

def _libreco_load(wrapper, p: Path) -> bool:
    data_info = getattr(wrapper, "data_info", None)
    if data_info is None:
        print("[adapters.py] ERROR: _libreco_load: wrapper.data_info is None.")
        return False

    model_class = getattr(wrapper, "model_type", None)
    if model_class is None:
        print("[adapters.py] ERROR: _libreco_load: wrapper.model_type is missing.")
        return False

    path_dir = str(p.parent)
    model_name = p.name
    print(f"[adapters.py] Loading LibReCo from {path_dir} (name={model_name})...")
    try:
        wrapper.model = model_class.load(path=path_dir, model_name=model_name, data_info=data_info, manual=True)
        return wrapper.model is not None
    except Exception as e:
        print(f"[adapters.py] ERROR: Failed to load LibReCo: {e}")
        return False


def save_model_auto(model_name: str, recs_model, run_dir: Path | str) -> str:
    pdir = Path(run_dir)
    pdir.mkdir(parents=True, exist_ok=True)

    key = _normalize_key(model_name)
    primary: Path | None = None

    try:
        import torch  
        if hasattr(recs_model, "state_dict") or (
            hasattr(recs_model, "model") and hasattr(recs_model.model, "state_dict")
        ):
            out = pdir / "last.pt"
            _torch_save(recs_model, out)
            primary = out
    except ImportError:
        pass
    except Exception:
        pass

    if primary is None:
        try:
            import tensorflow as tf  
            if hasattr(recs_model, "model") and hasattr(recs_model.model, "save_weights"):
                out = pdir / "keras_weights"
                _keras_save(recs_model.model, out)
                primary = out.with_suffix(".weights.h5")
        except ImportError:
            pass
        except Exception:
            pass

    if primary is None:
        if key == "lightfm":
            out = pdir / "lightfm"
            _lightfm_save(recs_model, out)
            primary = out.with_suffix(".npz")
        elif key == "als":
            if hasattr(recs_model, "U") and hasattr(recs_model, "V"):
                out = pdir / "als_factors.npz"
                _als_save(recs_model, out)
                primary = out
        elif key == "nmf":
            if hasattr(recs_model, "_W") and hasattr(recs_model, "_H"):
                out = pdir / "nmf_factors.npz"
                _nmf_save(recs_model, out)
                primary = out
        elif key == "deepfm_libreco":
            out = pdir / "libreco_model"
            _libreco_save(recs_model, out)
            primary = out

    wrapper_path = pdir / "model.pkl"
    try:
        joblib.dump(recs_model, wrapper_path, compress=3)
        if primary is None:
            primary = wrapper_path
    except Exception:
        pass

    if primary is None:
        print("[adapters.py] Final fallback: saving with joblib (model.pkl).")
        joblib.dump(recs_model, wrapper_path, compress=3)
        primary = wrapper_path

    return str(primary)

def load_model_auto(model_name: str, recs_model, run_dir: Path | str):
    pdir = Path(run_dir)
    key = _normalize_key(model_name)

    gp = pdir / "model.pkl"
    if _exists(gp):
        try:
            return joblib.load(gp)
        except Exception:
            pass

    if key == "lightfm":
        base = pdir / "lightfm"
        if _exists(base.with_suffix(".npz")) and _lightfm_load(recs_model, base):
            return True
    elif key == "als":
        als_path = pdir / "als_factors.npz"
        if _exists(als_path) and _als_load(recs_model, als_path):
            return True
    elif key == "nmf":
        nmf_path = pdir / "nmf_factors.npz"
        if _exists(nmf_path) and _nmf_load(recs_model, nmf_path):
            return True
    elif key == "deepfm_libreco":
        has_any = any(pdir.glob("libreco_model*"))
        if has_any and _libreco_load(recs_model, pdir / "libreco_model"):
            return True

    pt = pdir / "last.pt"
    try:
        import torch  
        if _exists(pt):
            if _torch_load(recs_model, pt):
                return True
    except ImportError:
        pass
    except Exception:
        pass

    try:
        import tensorflow as tf  
        kw = pdir / "keras_weights"
        if _exists(kw.with_suffix(".weights.h5")) and _keras_load(recs_model, kw):
            return True
        km_dir = pdir / "keras_model"
        if km_dir.exists() and km_dir.is_dir():
            try:
                from tensorflow import keras
                obj = type("KerasWrap", (), {})()
                obj.model = keras.models.load_model(str(km_dir))
                return obj
            except Exception:
                pass
    except ImportError:
        pass
    except Exception:
        pass

    return None
