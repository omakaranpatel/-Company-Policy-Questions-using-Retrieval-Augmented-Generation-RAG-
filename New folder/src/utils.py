"""Utility functions and Windows Streamlit stream patches."""

import os
import sys


def patch_sys_std_streams() -> None:
    """Safely patch sys.stderr.flush, sys.stdout.flush, and tqdm status_printer

    to prevent OSError: [Errno 22] Invalid argument in Windows Streamlit execution threads.
    """
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for stream_name in ("stderr", "stdout", "__stderr__", "__stdout__"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "flush"):
            try:
                orig_flush = stream.flush

                def make_safe(f):
                    def safe_flush():
                        try:
                            f()
                        except Exception:
                            pass
                    return safe_flush

                stream.flush = make_safe(orig_flush)
            except Exception:
                pass

    try:
        import tqdm.std
        _orig_sp = tqdm.std.tqdm.status_printer

        @staticmethod
        def _safe_sp(file):
            sp = _orig_sp(file)
            def _printer(str_to_print=""):
                try:
                    sp(str_to_print)
                except Exception:
                    pass
            return _printer

        tqdm.std.tqdm.status_printer = _safe_sp
    except Exception:
        pass

    try:
        from transformers import logging as tf_logging
        tf_logging.set_verbosity_error()
        tf_logging.disable_progress_bar()
    except Exception:
        pass


# Execute stream patch immediately upon import
patch_sys_std_streams()
