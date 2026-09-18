"""Collection lifecycle helpers, independent of Isaac Sim."""

import faulthandler
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path


class ManipulationFailure(RuntimeError):
    """A rejected scene or unsuccessful physical demonstration that may be retried."""


class CollectionStopped(RuntimeError):
    """An orderly stop requested by the operator."""


def is_incomplete_dataset_stub(root) -> bool:
    """Detect a dataset directory left behind before the first episode saved.

    LeRobot writes ``meta/info.json`` at creation but only writes
    ``meta/tasks.parquet`` on the first ``save_episode()``. A directory with
    zero recorded episodes and frames and no tasks file therefore holds no
    data and is safe to remove and recreate. Anything else missing the tasks
    file is genuine corruption that needs operator attention, and unreadable
    metadata returns False so the normal open path surfaces the real error.
    """
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not root.exists() or not info_path.is_file():
        return False
    if (root / "meta" / "tasks.parquet").is_file():
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    try:
        episodes = int(info.get("total_episodes", 0))
        frames = int(info.get("total_frames", 0))
    except (TypeError, ValueError):
        return False
    return episodes == 0 and frames == 0


class StopRequest:
    """Defer SIGINT/SIGTERM until a safe point, particularly during episode saving."""

    def __init__(self):
        self.signum = None
        self._previous = {}

    def install(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.signal(signum, self._request)

    def _request(self, signum, _frame):
        self.signum = signum

    def check(self):
        if self.signum is not None:
            raise CollectionStopped(f"Stop requested by {signal.Signals(self.signum).name}")

    def restore(self):
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


def _encoder_threads():
    """Return live video-encoder worker threads without importing LeRobot."""
    return [
        thread
        for thread in threading.enumerate()
        if type(thread).__name__ == "_CameraEncoderThread" and thread.is_alive()
    ]


def ensure_encoders_stopped(dataset, *, timeout_seconds=60.0, logger=None):
    """Cancel any active encoding episode and verify workers actually stopped.

    LeRobot's ``cancel_episode`` signals workers and joins them briefly but
    never verifies the outcome, so a stuck worker would otherwise surface
    later as a hang in ``finish_episode`` (up to 120 s per join) or as
    mysterious post-shutdown output. This helper bounds the wait, dumps the
    stacks of stuck workers for diagnosis, and raises so the caller can
    record the failure while still proceeding with finalize/close.
    """
    log = logger or logging.getLogger("ur10e.collection")
    encoder = getattr(dataset, "_streaming_encoder", None)
    if encoder is None:
        return True
    if getattr(encoder, "_episode_active", False):
        try:
            encoder.cancel_episode()
        except Exception:
            log.exception("Could not cancel active video encoding episode")
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while time.monotonic() < deadline:
        if not _encoder_threads():
            return True
        time.sleep(0.2)
    stuck = _encoder_threads()
    frames = sys._current_frames()
    for thread in stuck:
        frame = frames.get(thread.ident)
        stack = (
            "".join(traceback.format_stack(frame))
            if frame is not None
            else "<thread exited while dumping>\n"
        )
        log.error(
            "Video encoder thread still alive after %.1fs: %s\n%s",
            timeout_seconds,
            thread.name,
            stack,
        )
    raise RuntimeError(
        f"{len(stuck)} video encoder thread(s) did not stop within {timeout_seconds}s"
    )


def _watchdog_expired(dump_file, exit_code, _exit=os._exit):
    """Dump all thread stacks, then terminate: shutdown itself is stuck."""
    if dump_file is not None:
        try:
            with open(dump_file, "ab") as handle:
                faulthandler.dump_traceback(file=handle)
                handle.write(
                    f"\nShutdown watchdog expired; terminating with exit code {exit_code}\n".encode()
                )
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            pass
    _exit(exit_code)


def close_with_watchdog(close_app, *, timeout_seconds, dump_file, exit_code, logger=None):
    """Run simulator shutdown, guaranteeing the process cannot hang in it.

    If ``close_app`` does not return within ``timeout_seconds``, all thread
    stacks are dumped to ``dump_file`` for post-mortem diagnosis and the
    process is terminated with ``exit_code``. All durable work (dataset
    finalize, statistics, logs) must already be finished before this call.
    A non-positive timeout disables the watchdog and calls ``close_app``.
    """
    log = logger or logging.getLogger("ur10e.collection")
    if timeout_seconds is None or float(timeout_seconds) <= 0:
        close_app()
        return
    finished = threading.Event()

    def _watch():
        if not finished.wait(float(timeout_seconds)):
            log.error(
                "Simulator shutdown did not finish within %.0fs; dumping stacks to %s",
                timeout_seconds,
                dump_file,
            )
            _watchdog_expired(str(dump_file), int(exit_code))

    watcher = threading.Thread(target=_watch, name="shutdown-watchdog", daemon=True)
    watcher.start()
    try:
        close_app()
    finally:
        finished.set()
        watcher.join(timeout=5)


def finalize_collection(
    dataset,
    *,
    discard,
    generate_stats,
    close_app,
    logger,
    shutdown_watchdog_seconds=None,
    watchdog_dump_file=None,
    exit_code=0,
    encoder_stop_timeout_seconds=60.0,
):
    """Finish all disk work before Kit shutdown, which may terminate Python."""
    errors = []
    finalized = False
    if dataset is not None:
        if discard:
            logger.info("Discarding incomplete episode buffer")
            try:
                dataset.clear_episode_buffer()
            except Exception as error:
                logger.exception("Could not discard incomplete episode")
                errors.append(error)
        logger.info("Verifying video encoder threads have stopped")
        try:
            ensure_encoders_stopped(
                dataset, timeout_seconds=encoder_stop_timeout_seconds, logger=logger
            )
        except Exception as error:
            logger.exception("Video encoder shutdown was not clean")
            errors.append(error)
        logger.info("Finalizing dataset")
        try:
            dataset.finalize()
            finalized = True
        except Exception as error:
            logger.exception("Could not finalize dataset")
            errors.append(error)
        if finalized and dataset.num_episodes > 0:
            logger.info("Generating dataset statistics")
            try:
                generate_stats()
            except Exception as error:
                logger.exception("Could not generate dataset statistics")
                errors.append(error)
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    logger.info("Closing simulator")
    try:
        close_with_watchdog(
            close_app,
            timeout_seconds=shutdown_watchdog_seconds,
            dump_file=watchdog_dump_file,
            exit_code=exit_code,
            logger=logger,
        )
        logger.info("Simulator closed")
    except Exception as error:
        logger.exception("Could not shut down simulator")
        errors.append(error)
    if errors:
        raise RuntimeError("Collection cleanup failed; see collection log") from errors[0]
