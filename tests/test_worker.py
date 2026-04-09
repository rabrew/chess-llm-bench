"""Tests for src/worker.py"""

import pytest
from unittest.mock import MagicMock, patch, call

from src.worker import Worker, run_worker
from src.job_queue import JobQueue
from src.llm_client import OllamaClient
from src.result_writer import ResultWriter
from src.data_loader import DataLoader


FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"

SAMPLE_POSITION = {
    "id": 1,
    "fen": FEN,
    "pgn_moves": "1. e4 e5 2. Nf3 Nc6 3. Bc4",
    "difficulty": "medium",
    "phase": "opening",
    "source": "lichess_puzzles",
    "theme": "pin",
    "stockfish_eval": 50,
    "stockfish_best_move": "Nf6",
}

SAMPLE_JOB = {
    "job_id": "job_00001_llama3_2_3b_pgn+fen_1",
    "job_type": "standard",
    "position_id": 1,
    "model": "llama3.2:3b",
    "prompt_format": "pgn+fen",
    "trial": 1,
}

LLM_LEGAL_RESPONSE = {
    "response": "Eval: 45\nMove: Nf6\nExplanation: Equal — Both sides equal.",
    "inference_ms": 1500,
    "success": True,
    "model": "llama3.2:3b",
}

LLM_FAILED_RESPONSE = {
    "response": "",
    "inference_ms": 100,
    "success": False,
    "error": "timeout",
    "model": "llama3.2:3b",
}


@pytest.fixture
def worker_config(tmp_path):
    db_path = str(tmp_path / "jobs.db")
    results_path = str(tmp_path / "results" / "evaluations.jsonl")
    data_dir = str(tmp_path / "data")
    return {
        "paths": {
            "jobs_db": db_path,
            "results_file": results_path,
            "data_dir": data_dir,
        },
        "ollama": {"base_url": "http://localhost:11434", "timeout": 180, "max_retries": 1},
        "evaluation": {
            "centipawn_eval_range": {"min": -2000, "max": 2000},
            "cpl_threshold": 50,
        },
        "correction_loop": {"enabled": False},
        "benchmark": {"random_seed": 42},
    }


@pytest.fixture
def worker(worker_config):
    with patch("src.worker.OllamaClient"), \
         patch("src.worker.DataLoader"), \
         patch("src.worker.ResultWriter"), \
         patch("src.worker.get_completed_job_ids", return_value=set()):
        w = Worker("worker_1", worker_config)
    return w


class TestWorkerInit:
    def test_creates_worker(self, worker_config):
        with patch("src.worker.OllamaClient"), \
             patch("src.worker.DataLoader"), \
             patch("src.worker.ResultWriter"), \
             patch("src.worker.get_completed_job_ids", return_value=set()):
            w = Worker("w1", worker_config)
        assert w.worker_id == "w1"
        assert w.dry_run is False

    def test_dry_run_flag(self, worker_config):
        with patch("src.worker.OllamaClient"), \
             patch("src.worker.DataLoader"), \
             patch("src.worker.ResultWriter"), \
             patch("src.worker.get_completed_job_ids", return_value=set()):
            w = Worker("w1", worker_config, dry_run=True)
        assert w.dry_run is True


class TestProcessJob:
    def _setup_worker(self, worker_config, position=SAMPLE_POSITION,
                      completed=None, llm_response=None):
        """Create a fully mocked worker."""
        if completed is None:
            completed = set()
        if llm_response is None:
            llm_response = LLM_LEGAL_RESPONSE

        with patch("src.worker.OllamaClient") as mock_client_cls, \
             patch("src.worker.DataLoader") as mock_loader_cls, \
             patch("src.worker.ResultWriter") as mock_writer_cls, \
             patch("src.worker.get_completed_job_ids", return_value=completed):

            mock_client = MagicMock()
            mock_client.chat.return_value = llm_response
            mock_client_cls.return_value = mock_client

            mock_loader = MagicMock()
            mock_loader.get_by_id.return_value = position
            mock_loader_cls.return_value = mock_loader

            mock_writer = MagicMock()
            mock_writer_cls.return_value = mock_writer

            w = Worker("worker_test", worker_config)
            w.llm_client = mock_client
            w.data_loader = mock_loader
            w.result_writer = mock_writer
            w.job_queue = MagicMock()
            w.correction_manager = MagicMock()

        return w

    def test_skips_already_completed_job(self, worker_config):
        job_id = SAMPLE_JOB["job_id"]
        w = self._setup_worker(worker_config, completed={job_id})
        result = w.process_job(SAMPLE_JOB)
        assert result is None
        w.job_queue.complete_job.assert_called_once_with(job_id)

    def test_fails_if_position_not_found(self, worker_config):
        w = self._setup_worker(worker_config, position=None)
        result = w.process_job(SAMPLE_JOB)
        assert result is None
        w.job_queue.fail_job.assert_called()

    def test_processes_pgn_fen_format(self, worker_config):
        w = self._setup_worker(worker_config)
        result = w.process_job(SAMPLE_JOB)
        assert result is not None
        assert result["job_id"] == SAMPLE_JOB["job_id"]

    def test_fails_on_llm_error(self, worker_config):
        w = self._setup_worker(worker_config, llm_response=LLM_FAILED_RESPONSE)
        result = w.process_job(SAMPLE_JOB)
        assert result is None
        w.job_queue.fail_job.assert_called()

    def test_eval_only_format(self, worker_config):
        job = {**SAMPLE_JOB, "prompt_format": "eval_only"}
        eval_response = {"response": "45", "inference_ms": 500, "success": True, "model": "llama3.2:3b"}
        w = self._setup_worker(worker_config, llm_response=eval_response)
        result = w.process_job(job)
        assert result is not None

    def test_eval_only_llm_failure(self, worker_config):
        job = {**SAMPLE_JOB, "prompt_format": "eval_only"}
        w = self._setup_worker(worker_config, llm_response=LLM_FAILED_RESPONSE)
        result = w.process_job(job)
        assert result is None

    def test_move_only_format(self, worker_config):
        job = {**SAMPLE_JOB, "prompt_format": "move_only"}
        move_response = {"response": "Nf6", "inference_ms": 300, "success": True, "model": "llama3.2:3b"}
        w = self._setup_worker(worker_config, llm_response=move_response)
        result = w.process_job(job)
        assert result is not None

    def test_move_only_llm_failure(self, worker_config):
        job = {**SAMPLE_JOB, "prompt_format": "move_only"}
        w = self._setup_worker(worker_config, llm_response=LLM_FAILED_RESPONSE)
        result = w.process_job(job)
        assert result is None

    def test_explanation_only_format(self, worker_config):
        job = {**SAMPLE_JOB, "prompt_format": "explanation_only"}
        expl_response = {
            "response": "Explanation: Equal — Both sides equal.",
            "inference_ms": 400, "success": True, "model": "llama3.2:3b"
        }
        w = self._setup_worker(worker_config, llm_response=expl_response)
        result = w.process_job(job)
        assert result is not None

    def test_explanation_only_llm_failure(self, worker_config):
        job = {**SAMPLE_JOB, "prompt_format": "explanation_only"}
        w = self._setup_worker(worker_config, llm_response=LLM_FAILED_RESPONSE)
        result = w.process_job(job)
        assert result is None

    def test_fen_only_format(self, worker_config):
        job = {**SAMPLE_JOB, "prompt_format": "fen_only"}
        w = self._setup_worker(worker_config)
        result = w.process_job(job)
        assert result is not None

    def test_combined_fails_with_all_none(self, worker_config):
        """Combined prompt returns empty response — all fields None => fail."""
        empty_response = {"response": "", "inference_ms": 200, "success": True, "model": "x"}
        # Move retry also fails
        w = self._setup_worker(worker_config, llm_response=empty_response)
        # Make both calls return empty
        w.llm_client.chat.return_value = {"response": "", "inference_ms": 100,
                                           "success": True, "model": "x"}
        result = w.process_job(SAMPLE_JOB)
        assert result is None

    def test_does_not_write_in_dry_run(self, worker_config):
        w = self._setup_worker(worker_config)
        w.dry_run = True
        result = w.process_job(SAMPLE_JOB)
        assert result is not None
        w.result_writer.write_result.assert_not_called()

    def test_correction_triggered_on_high_cpl(self, worker_config):
        """correction_loop enabled + high CPL should call trigger_correction."""
        worker_config["correction_loop"]["enabled"] = True
        # Use a response that produces a legal move that is NOT the best move
        w = self._setup_worker(worker_config)
        # Patch score_all to return high CPL
        with patch("src.worker.score_all") as mock_score, \
             patch("src.worker.should_trigger_correction", return_value=True):
            mock_score.return_value = {
                "t1_model_eval": 45, "t1_stockfish_eval": 50,
                "t1_absolute_error": 5, "t1_direction_correct": True,
                "t2_move": "Nf6", "t2_best_move": "Nf6", "t2_legal": True, "t2_cpl": 200,
                "t3_explanation": "Equal.", "t3_side_claimed": "Equal",
                "t3_p1_side_correct": 1, "t3_p2_theme_correct": 0, "t3_score": 1,
            }
            result = w.process_job(SAMPLE_JOB)
        w.correction_manager.trigger_correction.assert_called_once()


class TestWorkerRun:
    def test_run_no_jobs(self, worker_config):
        with patch("src.worker.OllamaClient") as mock_client_cls, \
             patch("src.worker.DataLoader"), \
             patch("src.worker.ResultWriter"), \
             patch("src.worker.get_completed_job_ids", return_value=set()):
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client_cls.return_value = mock_client

            w = Worker("w1", worker_config)
            w.job_queue = MagicMock()
            w.job_queue.claim_job.return_value = None

            count = w.run()
        assert count == 0

    def test_run_raises_if_ollama_unavailable(self, worker_config):
        with patch("src.worker.OllamaClient") as mock_client_cls, \
             patch("src.worker.DataLoader"), \
             patch("src.worker.ResultWriter"), \
             patch("src.worker.get_completed_job_ids", return_value=set()):
            mock_client = MagicMock()
            mock_client.is_available.return_value = False
            mock_client_cls.return_value = mock_client

            w = Worker("w1", worker_config)
        with pytest.raises(RuntimeError, match="not available"):
            w.run()

    def test_run_respects_max_jobs(self, worker_config):
        with patch("src.worker.OllamaClient") as mock_client_cls, \
             patch("src.worker.DataLoader") as mock_loader_cls, \
             patch("src.worker.ResultWriter"), \
             patch("src.worker.get_completed_job_ids", return_value=set()):
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.chat.return_value = LLM_LEGAL_RESPONSE
            mock_client_cls.return_value = mock_client

            mock_loader = MagicMock()
            mock_loader.get_by_id.return_value = SAMPLE_POSITION
            mock_loader_cls.return_value = mock_loader

            w = Worker("w1", worker_config)
            w.job_queue = MagicMock()
            w.job_queue.claim_job.return_value = {**SAMPLE_JOB}
            w.result_writer = MagicMock()
            w.correction_manager = MagicMock()

            count = w.run(max_jobs=2)
        assert count == 2

    def test_run_handles_process_job_exception(self, worker_config):
        with patch("src.worker.OllamaClient") as mock_client_cls, \
             patch("src.worker.DataLoader"), \
             patch("src.worker.ResultWriter"), \
             patch("src.worker.get_completed_job_ids", return_value=set()):
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client_cls.return_value = mock_client

            w = Worker("w1", worker_config)
            w.job_queue = MagicMock()
            # First call returns a job, second returns None
            w.job_queue.claim_job.side_effect = [{**SAMPLE_JOB}, None]
            w.job_queue.fail_job = MagicMock()

            with patch.object(w, "process_job", side_effect=RuntimeError("boom")):
                count = w.run()

        assert count == 0
        w.job_queue.fail_job.assert_called()


class TestRunWorker:
    def test_run_worker_function(self, worker_config, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(worker_config, f)

        with patch("src.worker.load_config", return_value=worker_config), \
             patch("src.worker.Worker") as mock_worker_cls:
            mock_worker = MagicMock()
            mock_worker.run.return_value = 5
            mock_worker_cls.return_value = mock_worker

            result = run_worker("w1", config_path=config_path, max_jobs=5)
        assert result == 5
