from pathlib import Path
from typing import Tuple, Any

from .adapters import save_model_auto, load_model_auto


def resume_if_exists(model_key: str, model: Any, run_dir: Path) -> Tuple[Any, bool]:
    try:
        loaded = load_model_auto(model_key, model, run_dir)

        if loaded is None:
            return model, False

        if isinstance(loaded, tuple) and len(loaded) == 2 and isinstance(loaded[1], bool):
            return loaded[0], loaded[1]

        return loaded, True

    except FileNotFoundError:
        return model, False
    except Exception as e:
        print(
            f"[checkpoint] Warning: failed to load checkpoint ({model_key}). "
            f"Continuing without resume. Error: {e.__class__.__name__}: {e}"
        )
        return model, False


def save_if_enabled(model_key: str, model: Any, run_dir: Path, enabled: bool = True) -> bool:
    if not enabled:
        return False

    try:
        save_model_auto(model_key, model, run_dir)
        return True

    except Exception as e:
        print(
            f"[checkpoint] Warning: could not save checkpoint ({model_key}). "
            f"Continuing without checkpoint. Error: {e.__class__.__name__}: {e}"
        )

        try:
            p = Path(run_dir) / "checkpoint_save_failed.txt"
            p.write_text(
                f"model_key={model_key}\nerror={e.__class__.__name__}: {e}\n",
                encoding="utf-8",
            )
        except Exception:
            pass

        return False
