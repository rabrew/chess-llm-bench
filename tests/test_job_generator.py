"""Tests for src/job_generator.py"""

import pytest
from unittest.mock import MagicMock, patch

from src.job_generator import (
    generate_job_id,
    generate_standard_jobs,
    generate_correction_jobs,
    populate_job_queue,
    estimate_job_count,
)


SAMPLE_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK22R b KQkq - 3 3"


@pytest.fixture
def sample_positions():
    return [
        {"id": 1, "fen": "fen_1", "difficulty": "easy", "phase": "opening",
         "source": "lichess", "theme": "pin", "pgn_moves": "1. e4"},
        {"id": 2, "fen": "fen_2", "difficulty": "medium", "phase": "middlegame",
         "source": "lichess", "theme": "fork", "pgn_moves": ""},
    ]


@pytest.fixture
def sample_models():
    return ["llama3.2:3b", "qwen2.5:7b"]


@pytest.fixture
def sample_formats():
    return ["fen_only", "pgn+fen"]


class TestGenerateJobId:
    def test_basic_format(self):
        job_id = generate_job_id(1, "qwen2.5:7b", "pgn+fen")
        assert "00001" in job_id
        assert "qwen2_5_7b" in job_id
        assert "pgn+fen" in job_id

    def test_trial_included(self):
        job_id = generate_job_id(5, "llama3.2:3b", "cot", trial=2)
        assert "00005" in job_id
        assert "2" in job_id

    def test_special_chars_replaced(self):
        job_id = generate_job_id(1, "model:tag", "fmt")
        assert ":" not in job_id
        assert "." not in job_id


class TestGenerateStandardJobs:
    def test_correct_count(self, sample_positions, sample_models, sample_formats):
        jobs = generate_standard_jobs(sample_positions, sample_models, sample_formats)
        # 2 positions × 2 models × 2 formats = 8
        assert len(jobs) == 8

    def test_job_structure(self, sample_positions, sample_models, sample_formats):
        jobs = generate_standard_jobs(sample_positions, sample_models, sample_formats)
        job = jobs[0]
        assert "job_id" in job
        assert "job_type" in job
        assert job["job_type"] == "standard"
        assert "hash" in job
        assert "position_id" in job
        assert "model" in job
        assert "prompt_format" in job

    def test_empty_positions(self, sample_models, sample_formats):
        jobs = generate_standard_jobs([], sample_models, sample_formats)
        assert jobs == []

    def test_position_metadata_preserved(self, sample_positions, sample_models, sample_formats):
        jobs = generate_standard_jobs(sample_positions, sample_models, sample_formats)
        job = jobs[0]
        assert job["difficulty"] in ["easy", "medium"]
        assert job["phase"] in ["opening", "middlegame"]


class TestGenerateCorrectionJobs:
    @pytest.fixture
    def parent_position(self):
        return {"id": 1, "fen": "fen_1", "theme": "pin", "difficulty": "easy"}

    @pytest.fixture
    def follow_up_position(self):
        return {"id": 2, "fen": "fen_2", "theme": "pin", "difficulty": "easy"}

    def test_returns_two_jobs(self, parent_position, follow_up_position):
        corr, ctrl = generate_correction_jobs(
            parent_position, "llama3.2:3b", "pgn+fen",
            "parent_job_001", follow_up_position
        )
        assert corr is not None
        assert ctrl is not None

    def test_correction_job_type(self, parent_position, follow_up_position):
        corr, ctrl = generate_correction_jobs(
            parent_position, "llama3.2:3b", "pgn+fen",
            "parent_job_001", follow_up_position
        )
        assert corr["job_type"] == "correction"
        assert ctrl["job_type"] == "control"

    def test_ids_derived_from_parent(self, parent_position, follow_up_position):
        corr, ctrl = generate_correction_jobs(
            parent_position, "llama3.2:3b", "pgn+fen",
            "parent_001", follow_up_position
        )
        assert corr["job_id"] == "parent_001_correction"
        assert ctrl["job_id"] == "parent_001_control"

    def test_cross_linked(self, parent_position, follow_up_position):
        corr, ctrl = generate_correction_jobs(
            parent_position, "llama3.2:3b", "pgn+fen",
            "p1", follow_up_position
        )
        assert corr["paired_control_job_id"] == ctrl["job_id"]
        assert ctrl["paired_control_job_id"] == corr["job_id"]

    def test_different_hashes(self, parent_position, follow_up_position):
        corr, ctrl = generate_correction_jobs(
            parent_position, "llama3.2:3b", "pgn+fen",
            "p1", follow_up_position
        )
        assert corr["hash"] != ctrl["hash"]

    def test_trial_is_2(self, parent_position, follow_up_position):
        corr, ctrl = generate_correction_jobs(
            parent_position, "llama3.2:3b", "pgn+fen",
            "p1", follow_up_position
        )
        assert corr["trial"] == 2
        assert ctrl["trial"] == 2


class TestPopulateJobQueue:
    def test_inserts_jobs(self, tmp_path, sample_positions, sample_models, sample_formats):
        from src.data_loader import DataLoader
        from src.job_queue import JobQueue

        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.load_tier.return_value = sample_positions
        mock_loader.sample.return_value = sample_positions

        db_path = str(tmp_path / "jobs.db")
        queue = JobQueue(db_path)

        config = {
            "models": sample_models,
            "benchmark": {"prompt_formats": sample_formats, "max_positions_per_tier": 0},
            "paths": {"data_dir": "data", "jobs_db": db_path},
        }

        inserted = populate_job_queue(config, data_loader=mock_loader, job_queue=queue)
        assert inserted == len(sample_positions) * len(sample_models) * len(sample_formats)

    def test_returns_zero_for_no_positions(self, tmp_path, sample_models, sample_formats):
        from src.data_loader import DataLoader
        from src.job_queue import JobQueue

        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.load_tier.return_value = []

        db_path = str(tmp_path / "jobs.db")
        queue = JobQueue(db_path)

        config = {
            "models": sample_models,
            "benchmark": {"prompt_formats": sample_formats},
            "paths": {"data_dir": "data", "jobs_db": db_path},
        }
        inserted = populate_job_queue(config, data_loader=mock_loader, job_queue=queue)
        assert inserted == 0

    def test_returns_zero_for_no_models(self, tmp_path, sample_positions, sample_formats):
        from src.data_loader import DataLoader
        from src.job_queue import JobQueue

        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.load_tier.return_value = sample_positions

        db_path = str(tmp_path / "jobs.db")
        queue = JobQueue(db_path)

        config = {
            "models": [],
            "benchmark": {"prompt_formats": sample_formats},
            "paths": {"data_dir": "data", "jobs_db": db_path},
        }
        inserted = populate_job_queue(config, data_loader=mock_loader, job_queue=queue)
        assert inserted == 0

    def test_tier_filter(self, tmp_path, sample_positions, sample_models, sample_formats):
        from src.data_loader import DataLoader
        from src.job_queue import JobQueue

        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.load_tier.return_value = [sample_positions[0]]

        db_path = str(tmp_path / "jobs.db")
        queue = JobQueue(db_path)

        config = {
            "models": sample_models,
            "benchmark": {"prompt_formats": sample_formats, "max_positions_per_tier": 0},
            "paths": {"data_dir": "data", "jobs_db": db_path},
        }
        inserted = populate_job_queue(
            config, data_loader=mock_loader, job_queue=queue, tier="easy"
        )
        assert inserted > 0
        # Only called for "easy"
        mock_loader.load_tier.assert_called_once_with("easy")

    def test_model_filter(self, tmp_path, sample_positions, sample_formats):
        from src.data_loader import DataLoader
        from src.job_queue import JobQueue

        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.load_tier.return_value = sample_positions

        db_path = str(tmp_path / "jobs.db")
        queue = JobQueue(db_path)

        config = {
            "models": ["llama3.2:3b", "qwen2.5:7b"],
            "benchmark": {"prompt_formats": sample_formats, "max_positions_per_tier": 0},
            "paths": {"data_dir": "data", "jobs_db": db_path},
        }
        inserted = populate_job_queue(
            config, data_loader=mock_loader, job_queue=queue, model="llama3.2:3b"
        )
        total = queue.count_total()
        # All jobs should be for llama3.2:3b only
        jobs = queue.get_jobs_by_status("pending")
        assert all(j["model"] == "llama3.2:3b" for j in jobs)

    def test_max_positions_per_tier(self, tmp_path, sample_positions, sample_models, sample_formats):
        from src.data_loader import DataLoader
        from src.job_queue import JobQueue

        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.load_tier.return_value = sample_positions
        # Each tier gets a unique position to avoid hash collisions across tiers
        base = {"phase": "opening", "source": "lichess", "theme": "pin", "pgn_moves": ""}
        mock_loader.sample.side_effect = [
            [{"id": 10, "fen": "fen_easy", "difficulty": "easy", **base}],
            [{"id": 20, "fen": "fen_med", "difficulty": "medium", **base}],
            [{"id": 30, "fen": "fen_hard", "difficulty": "hard", **base}],
            [{"id": 40, "fen": "fen_ext", "difficulty": "extreme", **base}],
        ]

        db_path = str(tmp_path / "jobs.db")
        queue = JobQueue(db_path)

        config = {
            "models": sample_models,
            "benchmark": {"prompt_formats": sample_formats, "max_positions_per_tier": 1},
            "paths": {"data_dir": "data", "jobs_db": db_path},
        }
        inserted = populate_job_queue(config, data_loader=mock_loader, job_queue=queue)
        # 4 tiers × 1 pos × 2 models × 2 formats
        assert inserted == 4 * 1 * 2 * 2


class TestEstimateJobCount:
    def test_returns_estimates(self, sample_positions, sample_models, sample_formats):
        from src.data_loader import DataLoader

        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.load_all.return_value = sample_positions

        config = {
            "models": sample_models,
            "benchmark": {"prompt_formats": sample_formats},
            "paths": {"data_dir": "data"},
        }
        result = estimate_job_count(config, data_loader=mock_loader)
        assert "positions" in result
        assert "standard_jobs" in result
        assert "total_estimate" in result
        assert result["positions"] == 2
        assert result["standard_jobs"] == 2 * 2 * 2
