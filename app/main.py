"""CLI entry: fetch games and stats, build dataset, write Parquet."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import (
    DATA_DIR,
    DEFAULT_SEASON,
    STATS_TABLES_DIR,
    default_live_season,
    games_parquet_path,
)
from app.data.fetch_games import fetch_games
from app.data.fetch_kalshi import DEFAULT_KALSHI_MAP_PATH, build_kalshi_features_for_games
from app.data.fetch_platoon_mlb import fetch_team_platoon_ops_mlb, platoon_ops_to_index
from app.data.fetch_stats import fetch_batting_stats, fetch_pitcher_stats
from app.data.normalize import fangraphs_team_abbrev
from app.features.build_features import build_dataset
from app.features.bullpen import compute_bullpen_features_for_games
from app.features.rolling import compute_sp_rolling_for_games
from app.storage.db import save_to_parquet


def _export_stats_tables(
    pitchers: pd.DataFrame,
    batting: pd.DataFrame,
    season: int,
    export_dir: str,
) -> None:
    d = Path(export_dir)
    d.mkdir(parents=True, exist_ok=True)
    save_to_parquet(pitchers, str(d / f"pitchers_{season}.parquet"))
    save_to_parquet(batting, str(d / f"team_batting_{season}.parquet"))
    logging.info(
        "Wrote %s / %s",
        (d / f"pitchers_{season}.parquet").resolve(),
        (d / f"team_batting_{season}.parquet").resolve(),
    )


def _fg_teams_in_schedule(games: pd.DataFrame) -> list[str]:
    s: set[str] = set()
    for _, row in games.iterrows():
        h = fangraphs_team_abbrev(str(row.get("home_team_name") or ""))
        a = fangraphs_team_abbrev(str(row.get("away_team_name") or ""))
        if h:
            s.add(h)
        if a:
            s.add(a)
    return sorted(s)


def _extended_frames(
    games: pd.DataFrame,
    season: int,
    *,
    platoon: bool,
    rolling: bool,
    bullpen: bool,
    kalshi: bool,
    kalshi_map: str | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    platoon_by_team: pd.DataFrame | None = None
    roll_df: pd.DataFrame | None = None
    pen_df: pd.DataFrame | None = None
    kalshi_df: pd.DataFrame | None = None

    if platoon:
        teams = _fg_teams_in_schedule(games)
        raw = fetch_team_platoon_ops_mlb(season, teams)
        platoon_by_team = platoon_ops_to_index(raw).set_index("Team")

    if rolling:
        roll_df = compute_sp_rolling_for_games(games, season)

    if bullpen:
        pen_df = compute_bullpen_features_for_games(games, season)

    if kalshi:
        mp = kalshi_map if kalshi_map else None
        kalshi_df = build_kalshi_features_for_games(games, map_path=mp)

    return platoon_by_team, roll_df, pen_df, kalshi_df


def _write_last_run(
    outputs: list[dict[str, object]],
    meta: dict[str, object],
) -> None:
    """
    ``outputs``: items with keys ``path`` (str), ``row_count`` (int), optional ``season`` (int).
    """
    norm: list[dict[str, object]] = []
    paths: list[str] = []
    for o in outputs:
        p = Path(str(o["path"])).resolve()
        paths.append(str(p))
        item: dict[str, object] = {"out_path": str(p), "row_count": int(o["row_count"])}
        if "season" in o and o["season"] is not None:
            item["season"] = int(o["season"])  # type: ignore[arg-type]
        norm.append(item)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": norm,
        "output_paths": paths,
        **meta,
    }
    p = DATA_DIR / "last_run.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote %s", p.resolve())


def _feature_flags(ns: argparse.Namespace) -> dict[str, object]:
    return {
        "final_only": bool(ns.final_only),
        "platoon": not ns.no_platoon_br,
        "rolling": not ns.no_rolling,
        "bullpen": not ns.no_bullpen,
        "kalshi": bool(ns.kalshi),
        "kalshi_map": getattr(ns, "kalshi_map", None),
    }


def _execute_season_window(
    start_date: str,
    end_date: str,
    season: int,
    out_path: str,
    export_stats_dir: str | None,
    *,
    final_only: bool,
    platoon: bool,
    rolling: bool,
    bullpen: bool,
    kalshi: bool,
    kalshi_map: str | None,
    add_stats_season: bool,
) -> int:
    logging.info("Date range %s — %s, season=%s", start_date, end_date, season)

    games = fetch_games(start_date, end_date)
    logging.info("Games rows: %s", len(games))

    pitchers = fetch_pitcher_stats(season)
    batting = fetch_batting_stats(season)
    if export_stats_dir:
        _export_stats_tables(pitchers, batting, season, export_stats_dir)

    platoon_bt, roll_df, pen_df, kalshi_df = _extended_frames(
        games,
        season,
        platoon=platoon,
        rolling=rolling,
        bullpen=bullpen,
        kalshi=kalshi,
        kalshi_map=kalshi_map,
    )

    dataset = build_dataset(
        games,
        pitchers,
        batting,
        final_only=final_only,
        platoon_by_team=platoon_bt,
        roll_df=roll_df,
        pen_df=pen_df,
        kalshi_df=kalshi_df,
    )
    logging.info("Dataset rows: %s", len(dataset))

    if not dataset.empty and add_stats_season:
        dataset = dataset.copy()
        dataset["stats_season"] = season

    save_to_parquet(dataset, out_path)
    logging.info("Wrote %s", Path(out_path).resolve())
    return int(len(dataset))


def run_live(
    season: int,
    stats_dir: str,
    *,
    final_only: bool = False,
    platoon: bool = True,
    rolling: bool = True,
    bullpen: bool = True,
    kalshi: bool = False,
    kalshi_map: str | None = None,
    write_last_run: bool = True,
) -> None:
    start_date = f"{season}-03-01"
    end_date = f"{season}-11-30"
    out_path = str(games_parquet_path(season))
    n = _execute_season_window(
        start_date,
        end_date,
        season,
        out_path,
        stats_dir,
        final_only=final_only,
        platoon=platoon,
        rolling=rolling,
        bullpen=bullpen,
        kalshi=kalshi,
        kalshi_map=kalshi_map,
        add_stats_season=True,
    )
    if write_last_run:
        _write_last_run(
            [{"path": out_path, "row_count": n, "season": season}],
            {
                "subcommand": "live",
                "seasons": [season],
                "final_only": final_only,
                "platoon_mlb": platoon,
                "rolling": rolling,
                "bullpen": bullpen,
                "kalshi": kalshi,
                "stats_dir": str(Path(stats_dir).resolve()),
            },
        )


def run_backfill(
    years: list[int],
    stats_dir: str,
    *,
    final_only: bool = False,
    platoon: bool = True,
    rolling: bool = True,
    bullpen: bool = True,
    kalshi: bool = False,
    kalshi_map: str | None = None,
    write_last_run: bool = True,
) -> None:
    years_sorted = sorted(set(years))
    outputs: list[dict[str, object]] = []
    for y in years_sorted:
        start_date = f"{y}-03-01"
        end_date = f"{y}-11-30"
        out_path = str(games_parquet_path(y))
        logging.info("Backfill season %s: %s — %s", y, start_date, end_date)
        n = _execute_season_window(
            start_date,
            end_date,
            y,
            out_path,
            stats_dir,
            final_only=final_only,
            platoon=platoon,
            rolling=rolling,
            bullpen=bullpen,
            kalshi=kalshi,
            kalshi_map=kalshi_map,
            add_stats_season=True,
        )
        outputs.append({"path": out_path, "row_count": n, "season": y})

    if not outputs:
        raise RuntimeError("No seasons produced output; check years or API availability.")
    if write_last_run:
        _write_last_run(
            outputs,
            {
                "subcommand": "backfill",
                "seasons": years_sorted,
                "final_only": final_only,
                "platoon_mlb": platoon,
                "rolling": rolling,
                "bullpen": bullpen,
                "kalshi": kalshi,
                "stats_dir": str(Path(stats_dir).resolve()),
            },
        )


def run_custom(
    start_date: str,
    end_date: str,
    season: int,
    out_path: str,
    export_stats_dir: str | None,
    *,
    final_only: bool = False,
    platoon: bool = True,
    rolling: bool = True,
    bullpen: bool = True,
    kalshi: bool = False,
    kalshi_map: str | None = None,
    write_last_run: bool = True,
) -> None:
    n = _execute_season_window(
        start_date,
        end_date,
        season,
        out_path,
        export_stats_dir,
        final_only=final_only,
        platoon=platoon,
        rolling=rolling,
        bullpen=bullpen,
        kalshi=kalshi,
        kalshi_map=kalshi_map,
        add_stats_season=False,
    )
    if write_last_run:
        _write_last_run(
            [{"path": out_path, "row_count": n}],
            {
                "subcommand": "run",
                "season": season,
                "start_date": start_date,
                "end_date": end_date,
                "final_only": final_only,
                "platoon_mlb": platoon,
                "rolling": rolling,
                "bullpen": bullpen,
                "kalshi": kalshi,
                "export_stats_dir": str(Path(export_stats_dir).resolve()) if export_stats_dir else None,
            },
        )


def _add_shared_feature_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging",
    )
    p.add_argument(
        "--final-only",
        action="store_true",
        help="Only finished games (home_win is 0 or 1).",
    )
    p.add_argument(
        "--no-platoon-br",
        action="store_true",
        help="Skip platoon split columns (MLB Stats API source).",
    )
    p.add_argument(
        "--no-rolling",
        action="store_true",
        help="Skip 14-day rolling SP features.",
    )
    p.add_argument(
        "--no-bullpen",
        action="store_true",
        help="Skip bullpen season / 14-day features.",
    )
    p.add_argument(
        "--kalshi",
        action="store_true",
        help="Fetch Kalshi implieds (needs kalshi_game_map rows).",
    )
    p.add_argument(
        "--kalshi-map",
        metavar="CSV",
        default=None,
        help=f"game_pk to Kalshi ticker CSV (default {DEFAULT_KALSHI_MAP_PATH}).",
    )
    p.add_argument(
        "--no-last-run",
        action="store_true",
        help="Do not write data/last_run.json.",
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLB game + stats pipeline to Parquet",
        epilog=(
            "Examples:\n"
            "  python -m app.main live\n"
            "  python -m app.main live --season 2026\n"
            "  python -m app.main backfill --years 2024 2025\n"
            "  python -m app.main run --start 2024-04-01 --end 2024-04-30 --season 2024 --out data/sample.parquet\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    live_p = sub.add_parser("live", help="Sync current season: games_{season}.parquet + stats_tables/")
    _add_shared_feature_flags(live_p)
    live_p.add_argument(
        "--season",
        type=int,
        default=None,
        metavar="YEAR",
        help=f"Season year (default: PIPELINE_SEASON env or calendar year, currently would be {default_live_season()})",
    )
    live_p.add_argument(
        "--stats-dir",
        default=str(STATS_TABLES_DIR),
        metavar="DIR",
        help=f"Directory for pitchers_Y.parquet and team_batting_Y.parquet (default: {STATS_TABLES_DIR})",
    )

    bf_p = sub.add_parser("backfill", help="One file per season: games_Y.parquet + stats_tables/ for each year")
    _add_shared_feature_flags(bf_p)
    bf_p.add_argument(
        "--years",
        nargs="+",
        type=int,
        metavar="YEAR",
        required=True,
        help="Season years to build (Mar-Nov window each)",
    )
    bf_p.add_argument(
        "--stats-dir",
        default=str(STATS_TABLES_DIR),
        metavar="DIR",
        help=f"Directory for pitcher/team stat Parquet files (default: {STATS_TABLES_DIR})",
    )

    run_p = sub.add_parser(
        "run",
        help="Custom date window and output path (optional --export-stats-dir)",
    )
    _add_shared_feature_flags(run_p)
    run_p.add_argument("--start", required=True, help="Schedule start YYYY-MM-DD")
    run_p.add_argument("--end", required=True, help="Schedule end YYYY-MM-DD (inclusive)")
    run_p.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
        help=f"Stats season year (default {DEFAULT_SEASON})",
    )
    run_p.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Output games Parquet path",
    )
    run_p.add_argument(
        "--export-stats-dir",
        default=None,
        metavar="DIR",
        help="If set, also write pitchers_{season}.parquet and team_batting_{season}.parquet here",
    )

    return p.parse_args(argv)


def _try_legacy_cli(argv: list[str] | None) -> argparse.Namespace | None:
    """Support deprecated top-level --years / --start without subcommand."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "live", "backfill", "run"):
        return None
    if argv[0] not in ("--years", "--start", "--end", "--season", "--out"):
        return None
    logging.warning(
        "Invoking without a subcommand is deprecated. Use: live | backfill | run (see --help)."
    )
    legacy = argparse.ArgumentParser(description="MLB pipeline (deprecated)")
    legacy.add_argument("--start")
    legacy.add_argument("--end")
    legacy.add_argument("--season", type=int, default=DEFAULT_SEASON)
    legacy.add_argument("--years", nargs="+", type=int)
    legacy.add_argument(
        "--out",
        default=str(games_parquet_path(DEFAULT_SEASON)),
        help="Output path (backfill: ignored; one file per year is written)",
    )
    legacy.add_argument("-v", "--verbose", action="store_true")
    legacy.add_argument("--final-only", action="store_true")
    legacy.add_argument("--export-stats-dir", default=str(STATS_TABLES_DIR))
    legacy.add_argument("--no-platoon-br", action="store_true")
    legacy.add_argument("--no-rolling", action="store_true")
    legacy.add_argument("--no-bullpen", action="store_true")
    legacy.add_argument("--kalshi", action="store_true")
    legacy.add_argument("--kalshi-map", default=None)
    legacy.add_argument("--no-last-run", action="store_true")
    ns = legacy.parse_args(argv)
    ns.command = "_legacy"
    return ns


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    legacy_ns = _try_legacy_cli(argv)
    if legacy_ns is not None:
        args = legacy_ns
        level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        platoon = not args.no_platoon_br
        rolling = not args.no_rolling
        bullpen = not args.no_bullpen
        kalshi = bool(args.kalshi)
        try:
            if args.years:
                if args.start or args.end:
                    logging.error("Use either --years or --start/--end, not both.")
                    return 1
                run_backfill(
                    list(args.years),
                    str(args.export_stats_dir or STATS_TABLES_DIR),
                    final_only=args.final_only,
                    platoon=platoon,
                    rolling=rolling,
                    bullpen=bullpen,
                    kalshi=kalshi,
                    kalshi_map=args.kalshi_map,
                    write_last_run=not args.no_last_run,
                )
            else:
                if not args.start or not args.end:
                    logging.error("Provide --start and --end, or use --years")
                    return 1
                run_custom(
                    args.start,
                    args.end,
                    args.season,
                    args.out,
                    args.export_stats_dir,
                    final_only=args.final_only,
                    platoon=platoon,
                    rolling=rolling,
                    bullpen=bullpen,
                    kalshi=kalshi,
                    kalshi_map=args.kalshi_map,
                    write_last_run=not args.no_last_run,
                )
        except Exception:
            logging.exception("Pipeline failed")
            return 1
        return 0

    args = _parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ff = _feature_flags(args)
    try:
        if args.command == "live":
            season = args.season if args.season is not None else default_live_season()
            run_live(
                season,
                args.stats_dir,
                final_only=ff["final_only"],  # type: ignore[arg-type]
                platoon=ff["platoon"],  # type: ignore[arg-type]
                rolling=ff["rolling"],  # type: ignore[arg-type]
                bullpen=ff["bullpen"],  # type: ignore[arg-type]
                kalshi=ff["kalshi"],  # type: ignore[arg-type]
                kalshi_map=ff["kalshi_map"],  # type: ignore[arg-type]
                write_last_run=not args.no_last_run,
            )
        elif args.command == "backfill":
            run_backfill(
                list(args.years),
                args.stats_dir,
                final_only=ff["final_only"],  # type: ignore[arg-type]
                platoon=ff["platoon"],  # type: ignore[arg-type]
                rolling=ff["rolling"],  # type: ignore[arg-type]
                bullpen=ff["bullpen"],  # type: ignore[arg-type]
                kalshi=ff["kalshi"],  # type: ignore[arg-type]
                kalshi_map=ff["kalshi_map"],  # type: ignore[arg-type]
                write_last_run=not args.no_last_run,
            )
        elif args.command == "run":
            run_custom(
                args.start,
                args.end,
                args.season,
                args.out,
                args.export_stats_dir,
                final_only=ff["final_only"],  # type: ignore[arg-type]
                platoon=ff["platoon"],  # type: ignore[arg-type]
                rolling=ff["rolling"],  # type: ignore[arg-type]
                bullpen=ff["bullpen"],  # type: ignore[arg-type]
                kalshi=ff["kalshi"],  # type: ignore[arg-type]
                kalshi_map=ff["kalshi_map"],  # type: ignore[arg-type]
                write_last_run=not args.no_last_run,
            )
        else:
            logging.error("Unknown command %s", args.command)
            return 1
    except Exception:
        logging.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
