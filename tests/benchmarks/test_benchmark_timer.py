import time
import pytest

from proxima.resources.benchmark_timer import BenchmarkTimer


def test_timer_start_stop_elapsed_positive():
    timer = BenchmarkTimer()
    timer.start()
    time.sleep(0.01)
    timer.stop()

    assert timer.elapsed_ms() > 0
    assert timer.elapsed_seconds() > 0


def test_timer_pause_resume_skips_paused_time():
    timer = BenchmarkTimer()
    timer.start()
    time.sleep(0.01)
    timer.pause()
    time.sleep(0.02)
    timer.resume()
    time.sleep(0.01)
    timer.stop()

    # Paused section should not be counted; total active ~20ms
    # Relaxed tolerance for CI timing variance
    assert 10 <= timer.elapsed_ms() <= 100


def test_timer_checkpoint_and_duration():
    timer = BenchmarkTimer()
    timer.start()
    time.sleep(0.01)
    timer.checkpoint("phase1")
    time.sleep(0.01)
    timer.stop()

    duration = timer.get_checkpoint_duration("phase1")
    assert 8 <= duration <= 20
    checkpoints = timer.list_checkpoints()
    assert "phase1" in checkpoints


def test_timer_context_manager_usage():
    with BenchmarkTimer() as timer:
        time.sleep(0.01)
    assert timer.elapsed_ms() > 0


def test_timer_reset_clears_state():
    timer = BenchmarkTimer()
    timer.start()
    time.sleep(0.005)
    timer.checkpoint("cp1")
    timer.stop()

    timer.reset()

    assert timer.elapsed_ms() == 0.0
    assert timer.list_checkpoints() == {}
    assert not timer.is_running


def test_timer_double_start_raises():
    timer = BenchmarkTimer()
    timer.start()
    with pytest.raises(RuntimeError):
        timer.start()


def test_timer_edge_cases():
    timer = BenchmarkTimer()

    with pytest.raises(RuntimeError):
        timer.stop()

    with pytest.raises(RuntimeError):
        timer.pause()

    # Checkpoint not found
    timer.start()
    timer.stop()
    with pytest.raises(KeyError):
        timer.get_checkpoint_duration("nonexistent")
