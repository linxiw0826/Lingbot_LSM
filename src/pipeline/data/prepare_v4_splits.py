"""prepare_v4_splits.py

Split metadata_all.csv (v4 dynamic dataset, 46 episodes) into four CSV files
that CSGOMultiClipDataset consumes at different training phases.

Usage:
    python prepare_v4_splits.py --dataset_dir /home/nvme02/Memory-dataset/v4_dynamic_all46
    python prepare_v4_splits.py --dataset_dir /home/nvme02/Memory-dataset/v4_dynamic_all46 --dry_run

Output files (written next to metadata_all.csv):
    metadata_exp_train.csv   -- Exp-phase train  : episode_idx in {2,4,5,6,7,8,9,10,11}
    metadata_exp_val.csv     -- Exp-phase val    : episode_idx in {1,3}
    metadata_full_train.csv  -- Full-phase train : episode_idx in {15,17,...,46}
    metadata_full_val.csv    -- Full-phase val   : episode_idx in {12,13,14,16,21,22}
"""

import argparse
import collections
import csv
import logging
import os
import sys

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split definitions
# episode_idx values are integers; they are matched after int() conversion.
#
# 列语义说明：metadata_all.csv 中有两个不同列：
#   - episode_idx（整数，1-46）：episode 的全局编号，本脚本用于按 phase 过滤 clip 行
#   - episode_id（字符串，如 "player01_ep01"）：clip 的所属 episode 标识符，
#     下游训练脚本（CSGOMultiClipDataset._build_episode_groups）用于 clip 分组
# 二者功能不同，互不替代。
# ---------------------------------------------------------------------------

SPLIT_DEFS = collections.OrderedDict([
    (
        'metadata_exp_train.csv',
        {
            'episode_set': {2, 4, 5, 6, 7, 8, 9, 10, 11},
            'description': 'Exp-phase train (ep01-11, ~80%)',
        },
    ),
    (
        'metadata_exp_val.csv',
        {
            'episode_set': {1, 3},
            'description': 'Exp-phase val (ep01-11, ~20%)',
        },
    ),
    (
        'metadata_full_train.csv',
        {
            'episode_set': {
                15, 17, 18, 19, 20,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                38, 39, 40, 41, 42, 43, 44, 45, 46,
            },
            'description': 'Full-phase train',
        },
    ),
    (
        'metadata_full_val.csv',
        {
            'episode_set': {12, 13, 14, 16, 21, 22},
            'description': 'Full-phase val',
        },
    ),
])


def _read_csv(csv_path):
    """Read metadata_all.csv and return (fieldnames, rows).

    Args:
        csv_path (str): Absolute path to metadata_all.csv.

    Returns:
        tuple: (fieldnames: list[str], rows: list[dict])

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            'metadata_all.csv not found at: {}\n'
            'Make sure --dataset_dir points to the directory that contains '
            'metadata_all.csv.'.format(csv_path)
        )

    rows = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError('metadata_all.csv appears to be empty: {}'.format(csv_path))
        fieldnames = list(fieldnames)
        for row in reader:
            rows.append(row)

    logger.info('Read %d rows from %s', len(rows), csv_path)
    return fieldnames, rows


def _write_split(out_path, fieldnames, rows, split_name, episode_set, description, dry_run):
    """Filter rows by episode_set and write (or print) a split CSV.

    Args:
        out_path (str): Destination file path.
        fieldnames (list[str]): CSV column names (preserved from input).
        rows (list[dict]): All rows from metadata_all.csv.
        split_name (str): File name (used in log messages).
        episode_set (set[int]): episode_idx values to include.
        description (str): Human-readable description of the split.
        dry_run (bool): If True, only print stats; do not write any file.
    """
    filtered = [r for r in rows if int(r['episode_idx']) in episode_set]

    # Count distinct episodes actually present in the filtered set.
    present_episodes = {int(r['episode_idx']) for r in filtered}
    missing_episodes = episode_set - present_episodes

    if len(filtered) == 0:
        logger.warning(
            'WARNING: split "%s" (%s) has 0 rows. '
            'episode_set=%s — all episodes missing from CSV.',
            split_name, description, sorted(episode_set),
        )
    else:
        logger.info(
            'Split %-30s | %5d clips | %2d episodes | %s',
            split_name, len(filtered), len(present_episodes), description,
        )
        if missing_episodes:
            logger.warning(
                'Split "%s": expected episodes %s not found in CSV.',
                split_name, sorted(missing_episodes),
            )

    if dry_run:
        logger.info('[dry_run] Would write %d rows to %s', len(filtered), out_path)
        return

    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    logger.info('Wrote %s (%d rows)', out_path, len(filtered))


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Split metadata_all.csv into four phase-specific CSV files '
            'for CSGOMultiClipDataset training.'
        ),
    )
    parser.add_argument(
        '--dataset_dir',
        required=True,
        help='Root directory of the dataset. '
             'metadata_all.csv must exist here; output CSVs are written here too.',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print statistics only; do not write any output files.',
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    csv_path = os.path.join(dataset_dir, 'metadata_all.csv')

    fieldnames, rows = _read_csv(csv_path)

    for filename, spec in SPLIT_DEFS.items():
        out_path = os.path.join(dataset_dir, filename)
        _write_split(
            out_path=out_path,
            fieldnames=fieldnames,
            rows=rows,
            split_name=filename,
            episode_set=spec['episode_set'],
            description=spec['description'],
            dry_run=args.dry_run,
        )

    if args.dry_run:
        logger.info('[dry_run] No files written.')
    else:
        logger.info('Done. All split CSVs written to %s', dataset_dir)


if __name__ == '__main__':
    sys.exit(main())
