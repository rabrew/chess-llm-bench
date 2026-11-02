"""Synthetic regressions for the experiment metric audit."""
import numpy as np
import pandas as pd
import pytest
from src import metrics
from src.utils import load_config, parse_model_info
from src.metric_audit import direction_baselines, direction_diagnostics, regime_split, cpl_diagnostics


def test_all_configured_tags():
    for tag in load_config()["models"]:
        metadata = parse_model_info(tag)
        assert metadata["family"] != "unknown" and metadata["size_b"] > 0
    assert parse_model_info("gemma4:e2b")["size_b"] == 2
    assert parse_model_info("mixtral:8x7b")["is_moe"]
    assert parse_model_info("codellama:34b")["family"] == "codellama"


@pytest.mark.parametrize("tag", ["newmodel:7b", "llama3:latest", "llama3:0b", "llama3"])
def test_unknown_metadata_raises(tag):
    with pytest.raises(ValueError):
        parse_model_info(tag)


def direction_frame():
    return pd.DataFrame({"model": ["llama3.2:3b"] * 6, "position_id": list(range(6)),
                         "t1_stockfish_eval": [-300, 300, 0, 0, 0, 0],
                         "t1_model_eval": [0] * 6,
                         "fen": ["8/8/8/8/8/8/8/K6k w - - 0 1"] * 6})


def test_direction_baselines_and_constant_kappa():
    frame = direction_frame()
    baseline = direction_baselines(frame)
    assert baseline.majority_baseline_acc.tolist() == pytest.approx([4 / 6] * 4)
    diagnostic = direction_diagnostics(frame)
    assert (diagnostic.loc[diagnostic.n > 0, "kappa"] == 0).all()
    assert diagnostic.loc[diagnostic.n > 0, "balanced_accuracy"].tolist() == pytest.approx([1 / 3] * 8)


def test_undefined_kappa_and_scale_invariance():
    frame = direction_frame()
    frame["t1_stockfish_eval"] = [1, 2, 3, 4, 5, 6]
    frame["t1_model_eval"] = [1, 2, 3, 4, 5, 6]
    original = direction_diagnostics(frame)
    frame["t1_model_eval"] *= 5
    np.testing.assert_allclose(direction_diagnostics(frame).spearman_rho, original.spearman_rho, equal_nan=True)
    constant = direction_frame().assign(t1_stockfish_eval=0)
    assert direction_diagnostics(constant).kappa.isna().all()


def test_regime_does_not_invent_deployment_boundary():
    frame = pd.DataFrame({"model": ["x"] * 3, "prompt_format": ["cot"] * 3,
                          "t2_move": ["Qa9", "Ka2", None], "t2_legal": [False, True, False],
                          "timestamp": ["2026-01-01", "2026-01-02", None]})
    split = regime_split(frame)
    assert split.n.sum() == 3
    assert set(split.regime) == {"pre_rescue_evidence", "unknown"}
    assert split.loc[split.regime == "unknown", "n"].iloc[0] == 2


def test_cpl_bucket_counts_and_seeded_intervals():
    frame = pd.DataFrame({"model": ["a"] * 5 + ["b"] * 5,
                          "position_id": list(range(5)) * 2,
                          "t2_cpl_clamped": [0, 50, 100, 500, 1000] * 2})
    quality, buckets = cpl_diagnostics(frame, replicates=50)
    assert (quality.blunder_rate == .2).all()
    assert quality.rank_tied.all()
    assert buckets.groupby("model")["n"].sum().tolist() == [5, 5]
    again, _ = cpl_diagnostics(frame, replicates=50)
    pd.testing.assert_frame_equal(quality, again)


def test_family_preserves_same_size_models():
    frame = pd.DataFrame({"model": ["llama3.1:8b", "llama3.2:8b"],
                          "model_family": ["llama"] * 2, "model_size_b": [8] * 2,
                          "job_id": ["a", "b"], "t1_absolute_error": [1, 2],
                          "t2_cpl": [1, 2], "t2_legal": [True, False], "t3_score": [0, 1],
                          "t2_cpl_clamped": [1, 2], "t3_score_v2": [1, 2]})
    aggregate = metrics.aggregate_by_model_family(frame)
    assert len(aggregate) == 2
    assert "t2_cpl_clamped" in aggregate and "t3_score_v2" in aggregate


def test_historical_cpl_cannot_reconstruct_endpoints(tmp_path):
    import json
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(dict(t1_stockfish_eval=10000, t2_cpl=500,
                                   fen="8/8/8/8/8/8/8/K6k w - - 0 1")) + "\n")
    derived = metrics.load_results_df(str(path), str(tmp_path))
    assert derived.t2_cpl_clamped.isna().all()
    assert derived.t2_wp_loss.isna().all()
    assert derived.t2_endpoint_status.tolist() == ["unavailable"]


def test_exact_endpoint_metrics():
    record = dict(t2_reference_eval_before=1500, t2_reference_eval_after=500,
                  t2_reference_white_to_move=True)
    assert metrics.derive_endpoint_metrics(record)["t2_cpl_clamped"] == 500
    assert metrics.derive_endpoint_metrics({**record, "t2_reference_eval_before": float("inf")})["t2_endpoint_status"] == "invalid"


def test_stale_learning_file_removed(tmp_path, monkeypatch):
    from tests.test_metrics import sample_df
    frame = sample_df.__wrapped__()
    (tmp_path / "learning_deltas.csv").write_text("stale")
    metrics.save_metrics(frame, str(tmp_path))
    assert not (tmp_path / "learning_deltas.csv").exists()
    import json
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "t2_legal_rate" not in summary["overall_metrics"]
    assert summary["provenance"]["analysis_git_sha"]
    assert len(pd.read_csv(tmp_path / "by_model_family.csv")) == 2


def test_hypotheses_use_corrected_metrics():
    from tests.test_metrics import sample_df
    frame = sample_df.__wrapped__().assign(t3_score_v2=1, t2_cpl_clamped=10,
                                         t1_abs_error_excl_mate=20)
    hypotheses = metrics.compute_hypothesis_tests(frame)
    assert hypotheses['H3']['primary_metric'] == 't3_score_v2'
    assert hypotheses['H4']['primary_metric'] == 't2_cpl_clamped'
    assert set(hypotheses['H4']['metrics']) == {'t1_abs_error_excl_mate', 't2_cpl_clamped', 't3_score_v2'}
    assert hypotheses['H4']['supported'] is None
    assert hypotheses['H4']['publication_status'] == 'insufficient_data'


def test_legality_denominator_includes_failed_move_prompt(tmp_path):
    import json
    records = [dict(model='llama3.2:3b', prompt_format='cot', t2_move=None, t2_legal=False),
               dict(model='llama3.2:3b', prompt_format='cot', t2_move='Ka2', t2_legal=True),
               dict(model='llama3.2:3b', prompt_format='eval_only', t2_move=None, t2_legal=False)]
    path = tmp_path / 'results.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in records))
    loaded = metrics.load_results_df(str(path))
    assert loaded.t2_legal_move_prompt.mean() == .5


def test_legacy_csv_columns_are_explicit(tmp_path):
    from tests.test_metrics import sample_df
    metrics.save_metrics(sample_df.__wrapped__(), str(tmp_path))
    family = pd.read_csv(tmp_path / 'by_model_family.csv')
    assert 't2_legal_legacy' in family
    assert 't2_legal' not in family
    assert 't3_score_legacy' in family


def test_empty_dataset_removes_stale_outputs(tmp_path):
    (tmp_path / 'learning_deltas.csv').write_text('stale')
    (tmp_path / 'summary.json').write_text('stale')
    metrics.save_metrics(pd.DataFrame(), str(tmp_path))
    assert not (tmp_path / 'learning_deltas.csv').exists()
    import json
    assert json.loads((tmp_path / 'summary.json').read_text())['total_jobs'] == 0


def test_regime_assistance_classification():
    frame = pd.DataFrame(dict(model=['a']*4, prompt_format=['cot','cot','move_only','eval_only'],
                              t2_rescue_enabled=[False,True,False,False], t2_legal=[True]*4))
    split = regime_split(frame)
    assert set(split.regime) == {'single_shot','harness_assisted','legal_list_assisted','not_applicable'}
    assert split.n.sum() == len(frame)


def test_bootstrap_resamples_positions_as_clusters():
    frame = pd.DataFrame(dict(model=['a']*6, position_id=[1,1,1,2,2,2],
                              t2_cpl_clamped=[0,0,0,1000,1000,1000]))
    quality, _ = cpl_diagnostics(frame, replicates=100)
    assert quality.t2_cpl_clamped_ci_low.iloc[0] == 0
    assert quality.t2_cpl_clamped_ci_high.iloc[0] == 1000


def test_prompt_applicability_masks_spurious_scores(tmp_path):
    import json
    path = tmp_path / 'records.jsonl'
    records = [dict(model='llama3.2:3b', prompt_format=fmt, t1_model_eval=10,
                    t1_stockfish_eval=20, t1_absolute_error=10, theme='mate',
                    t3_explanation='White mates', t3_p1_side_correct=1)
               for fmt in ('move_only','eval_only','explanation_only','cot')]
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))
    frame = metrics.load_results_df(str(path))
    assert frame.t1_model_eval.notna().tolist() == [False, True, False, True]
    assert frame.t3_score_v2.notna().tolist() == [False, False, True, True]


def test_json_missing_diagnostics_are_null(tmp_path):
    import json
    from tests.test_metrics import sample_df
    frame = sample_df.__wrapped__().assign(t1_abs_error_excl_mate=np.nan)
    metrics.save_metrics(frame, str(tmp_path))
    text = (tmp_path/'summary.json').read_text()
    assert 'NaN' not in text
    assert json.loads(text)['overall_metrics']['t1_mean_error_excl_mate'] is None


def test_audit_cli_requires_explicit_destination(tmp_path):
    import subprocess, sys
    process = subprocess.run([sys.executable, 'scripts/generate_audit_metrics.py'], capture_output=True, text=True)
    assert process.returncode == 2
    assert '--output-dir' in process.stderr


def test_audit_cli_writes_temporary_outputs(tmp_path):
    import json, subprocess, sys
    from tests.test_metrics import sample_df
    path = tmp_path / 'records.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in sample_df.__wrapped__().to_dict('records')))
    target = tmp_path / 'audit'
    process = subprocess.run([sys.executable, 'scripts/generate_audit_metrics.py',
                              '--results', str(path), '--output-dir', str(target)],
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    summary = json.loads((target/'summary.json').read_text())
    assert summary['total_jobs'] == 8
    assert len(summary['provenance']['analysis_config_snapshot']['source_sha256']) == 64
    assert summary['provenance']['collection']['status'] == 'unverified'


def test_failure_rate_ignores_unasked_prompts():
    frame = pd.DataFrame(dict(model=['x']*3, difficulty=['easy']*3,
                              t2_legal=[False, True, False],
                              t2_legal_move_prompt=[0., 1., np.nan]))
    assert metrics.calculate_hallucination_rate(frame).move_prompt_failure_rate.iloc[0] == .5


def test_missing_hypothesis_evidence_is_unavailable():
    from tests.test_metrics import sample_df
    frame = sample_df.__wrapped__()
    # Add a second dense size in the same family so H4 has a real comparison.
    other = frame.loc[frame.model == 'llama3.2:3b'].copy()
    other['model'] = 'llama3.1:8b'
    frame = pd.concat([frame, other], ignore_index=True)
    frame = frame.assign(t1_abs_error_excl_mate=20, t2_cpl_clamped=np.nan, t3_score_v2=1)
    hypotheses = metrics.compute_hypothesis_tests(frame)
    assert hypotheses['H2']['supported'] is None
    assert hypotheses['H2']['publication_status'] == 'insufficient_data'
    assert hypotheses['H2']['metrics']['clamped_cpl']['available_n'] == 0
    assert hypotheses['H4']['supported'] is None
    assert hypotheses['H4']['publication_status'] == 'insufficient_data'
    family = hypotheses['H4']['by_family']['llama']
    assert family['metrics']['t2_cpl_clamped'] is None
    assert family['available_n']['t2_cpl_clamped'] == {'smaller': 0, 'larger': 0}
    assert family['all_better'] is None


def test_measured_negative_h4_is_false():
    from tests.test_metrics import sample_df
    frame = sample_df.__wrapped__()
    other = frame.loc[frame.model == 'llama3.2:3b'].copy()
    other['model'] = 'llama3.1:8b'
    frame = pd.concat([frame, other], ignore_index=True)
    frame = frame.assign(t1_abs_error_excl_mate=20, t2_cpl_clamped=10, t3_score_v2=1)
    hypothesis = metrics.compute_hypothesis_tests(frame)['H4']
    assert hypothesis['supported'] is False
    assert hypothesis['publication_status'] == 'available'


def convention_record():
    return dict(model='llama3.2:3b', position_id=1, t2_legal=True,
                t2_move='e4', t2_best_move='e4', t2_best_move_zero_convention=True,
                t2_scoring_version='lc0_endpoints_v1_best_move_zero',
                t2_reference_engine='Lc0Engine', t2_reference_budget={'nodes':800},
                t2_reference_units='centipawns_white')


def test_best_move_convention_quality_denominator():
    convention = convention_record()
    measured = dict(model='llama3.2:3b', position_id=2, t2_legal=True,
                    t2_reference_eval_before=500, t2_reference_eval_after=-500,
                    t2_reference_white_to_move=True)
    records = [{**row, **metrics.derive_endpoint_metrics(row)} for row in (convention, measured)]
    quality, buckets = cpl_diagnostics(pd.DataFrame(records), replicates=10)
    assert quality.n.iloc[0] == 2
    assert quality.blunder_rate.iloc[0] == .5
    assert quality.convention_zero_n.iloc[0] == 1
    assert quality.measured_endpoint_n.iloc[0] == 1
    assert records[0]['t2_endpoint_status'] == 'best_move_zero_convention'
    assert 't2_reference_eval_after' not in records[0]
    assert buckets.n.sum() == 2


@pytest.mark.parametrize('updates', [dict(t2_legal=False), dict(t2_best_move_zero_convention='true'),
    dict(t2_scoring_version='old'), dict(t2_reference_engine=None),
    dict(t2_reference_budget={'nodes':None}), dict(t2_reference_budget={'nodes':True}),
    dict(t2_reference_units='unknown'), dict(t2_best_move='d4')])
def test_unproven_best_move_convention_rejected(updates):
    row = {**convention_record(), **updates}
    assert metrics.derive_endpoint_metrics(row)['t2_endpoint_status'] == 'unavailable'
