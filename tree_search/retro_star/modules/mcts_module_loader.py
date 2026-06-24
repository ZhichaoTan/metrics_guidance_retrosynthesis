"""Load MCTS module files without executing tree_search.mcts.__init__."""
import importlib.util
import os
import sys
import types


def _tree_search_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_mcts_modules_package():
    """Register tree_search.mcts.modules so relative imports in module files work."""
    root = _tree_search_root()
    if root not in sys.path:
        sys.path.insert(0, os.path.dirname(root))

    mcts_dir = os.path.join(root, "mcts")
    modules_dir = os.path.join(mcts_dir, "modules")

    if "tree_search.mcts.modules" not in sys.modules:
        tree_search = sys.modules.get("tree_search")
        if tree_search is None:
            tree_search = types.ModuleType("tree_search")
            tree_search.__path__ = [root]
            sys.modules["tree_search"] = tree_search

        mcts = types.ModuleType("tree_search.mcts")
        mcts.__path__ = [mcts_dir]
        sys.modules["tree_search.mcts"] = mcts

        modules = types.ModuleType("tree_search.mcts.modules")
        modules.__path__ = [modules_dir]
        sys.modules["tree_search.mcts.modules"] = modules


def _load_module_from_mcts_modules(module_name: str, filename: str):
    _ensure_mcts_modules_package()
    path = os.path.join(_tree_search_root(), "mcts", "modules", filename)
    full_name = f"tree_search.mcts.modules.{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_local_pricer_api_class():
    """Load and return the MCTS ``LocalPricerAPI`` class."""
    mod = _load_module_from_mcts_modules("local_pricer_api", "local_pricer_api.py")
    return mod.LocalPricerAPI


def load_local_expand_one_api_class():
    """Load and return the MCTS ``LocalExpandOneAPI`` class (with its expand-one controller)."""
    _load_module_from_mcts_modules("expand_one_controller", "expand_one_controller.py")
    mod = _load_module_from_mcts_modules("local_expand_one_api", "local_expand_one_api.py")
    return mod.LocalExpandOneAPI
