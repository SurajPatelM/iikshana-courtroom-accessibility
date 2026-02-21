"""
Schema and statistics generation using Great Expectations (per PDF §2.7).
Validates data schema and generates statistics; results written to data/processed/.
"""
import json
from pathlib import Path

import pandas as pd

from scripts.utils import get_logger, load_config, PROCESSED_DIR

logger = get_logger("run_great_expectations")

try:
    import great_expectations as gx
except ImportError:
    gx = None  # type: ignore


def build_dataframe_from_quality_report(quality_report_path: Path) -> pd.DataFrame:
    """Build DataFrame from quality_report.json (produced by validate_schema)."""
    with open(quality_report_path, encoding="utf-8") as f:
        data = json.load(f)
    reports = data.get("file_reports", [])
    if not reports:
        return pd.DataFrame()
    rows = []
    for r in reports:
        path = r.get("path", "")
        split = "unknown"
        for s in ("dev", "test", "holdout"):
            if s in path:
                split = s
                break
        rows.append({
            "path": path,
            "split": split,
            "duration_sec": r.get("duration_sec"),
            "sample_rate": r.get("sample_rate"),
            "channels": r.get("channels"),
            "format": r.get("format"),
        })
    return pd.DataFrame(rows)


def run_ge_validation_and_statistics(
    data_dir: Path | None = None,
    quality_report_path: Path | None = None,
    result_path: Path | None = None,
    statistics_path: Path | None = None,
) -> dict:
    """
    Use Great Expectations to validate schema and generate statistics.
    Reads quality_report.json, builds DataFrame, runs GE expectations, writes results and statistics.
    """
    if data_dir is None:
        data_dir = PROCESSED_DIR
    data_dir = Path(data_dir)
    if quality_report_path is None:
        quality_report_path = data_dir / "quality_report.json"
    if result_path is None:
        result_path = data_dir / "ge_validation_result.json"
    if statistics_path is None:
        statistics_path = data_dir / "data_statistics.json"

    if not quality_report_path.exists():
        logger.warning("No quality_report.json found; run validate_schema first. Writing empty GE result.")
        out = {"success": False, "reason": "quality_report.json not found", "statistics": {}}
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        with open(statistics_path, "w", encoding="utf-8") as f:
            json.dump({"statistics": {}}, f, indent=2)
        return out

    df = build_dataframe_from_quality_report(quality_report_path)
    if df.empty:
        logger.warning("No file reports in quality_report.json")
        out = {"success": True, "results": {}, "statistics": {}}
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        with open(statistics_path, "w", encoding="utf-8") as f:
            json.dump({"statistics": {}}, f, indent=2)
        return out

    cfg = load_config().get("validation", {})
    expected_sr = cfg.get("expected_sr", 16000)
    min_dur = cfg.get("min_duration_sec", 0.5)
    max_dur = cfg.get("max_duration_sec", 30.0)

    validation_results = {}
    success = True

    if gx is not None:
        try:
            context = gx.get_context(mode="ephemeral")
            # Data source -> dataframe asset -> batch definition (GE v1 API)
            data_source = context.data_sources.add_pandas(name="processed_data")
            data_asset = data_source.add_dataframe_asset(name="processed_frame")
            batch_definition = data_asset.add_batch_definition_whole_dataframe("processed_batch")
            batch_parameters = {"dataframe": df}

            # Expectation suite with same expectations: duration_sec, sample_rate, channels
            suite = gx.ExpectationSuite(name="processed_suite")
            suite = context.suites.add(suite)
            suite.add_expectation(gx.expectations.ExpectColumnToExist(column="duration_sec"))
            suite.add_expectation(gx.expectations.ExpectColumnToExist(column="sample_rate"))
            suite.add_expectation(gx.expectations.ExpectColumnToExist(column="channels"))
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="duration_sec", min_value=min_dur, max_value=max_dur
                )
            )
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column="sample_rate", min_value=8000, max_value=48000
                )
            )
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeInSet(column="channels", value_set=[1, 2])
            )
            suite.save()

            validation_definition = gx.ValidationDefinition(
                data=batch_definition, suite=suite, name="processed_data_validation"
            )
            validation_definition = context.validation_definitions.add(validation_definition)
            result = validation_definition.run(batch_parameters=batch_parameters)

            success = result.success
            run_id = getattr(getattr(result, "meta", None), "run_id", "") or getattr(result, "batch_id", "")
            stats = getattr(result, "statistics", {})
            results_list = getattr(result, "results", [])
            if results_list and hasattr(results_list[0], "to_json_dict"):
                results_list = [r.to_json_dict() for r in results_list]
            elif results_list and not isinstance(results_list[0], dict):
                results_list = [getattr(r, "__dict__", r) for r in results_list]
            validation_results = {
                "success": success,
                "run_id": str(run_id),
                "statistics": stats if isinstance(stats, dict) else {},
                "results": results_list,
            }
        except Exception as e:
            logger.exception("Great Expectations validation failed: %s", e)
            validation_results = {"success": False, "error": str(e), "results": []}
            success = False
    else:
        logger.warning("Great Expectations not installed; skipping GE validation.")
        validation_results = {"success": True, "note": "GE not installed"}

    # Statistics generation (per PDF: data schema and statistics)
    statistics = {}
    if not df.empty:
        for col in ["duration_sec", "sample_rate", "channels"]:
            if col in df.columns:
                statistics[col] = {
                    "count": int(df[col].count()),
                    "mean": float(df[col].mean()) if df[col].dtype in ("float64", "int64") else None,
                    "std": float(df[col].std()) if df[col].dtype in ("float64", "int64") else None,
                    "min": float(df[col].min()) if df[col].dtype in ("float64", "int64") else None,
                    "max": float(df[col].max()) if df[col].dtype in ("float64", "int64") else None,
                }
        if "split" in df.columns:
            statistics["split_counts"] = df["split"].value_counts().to_dict()

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=2)
    with open(statistics_path, "w", encoding="utf-8") as f:
        json.dump({"statistics": statistics}, f, indent=2)

    logger.info("Great Expectations validation success=%s; statistics written to %s", success, statistics_path)
    return {"success": success, "validation_results": validation_results, "statistics": statistics}


def main() -> None:
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out = run_ge_validation_and_statistics(data_dir=data_dir)
    if not out.get("success", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
