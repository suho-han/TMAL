import importlib

_REGISTRY = {}


def register(name):
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator


def build_model(name, args):
    if name not in _REGISTRY:
        try:
            importlib.import_module(f'model.{name}')
        except ModuleNotFoundError as e:
            if e.name and not e.name.startswith(f'model.{name}'):
                raise ImportError(
                    f"model.{name} failed to import due to missing dependency: {e}"
                ) from e
        except ImportError as e:
            raise ImportError(f"model.{name} failed to import: {e}") from e
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](args)
