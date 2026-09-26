#!/usr/bin/env python3
"""Make a sampled baseline-development ZIP for the Amazon business entity resolution task.

Python 3.9+, standard library only. Reads an extracted folder or a ZIP directly.
Notebook: %run make_er_sample.py --input "student_resource.zip"
Terminal: python make_er_sample.py --input "student_resource.zip"

The originals are read only. All known matches for sampled training entities
are retained. Independently sampled test rows have NO completeness guarantee.
Defaults: 5,000 S1 rows per country, all their known matches, plus 5,000
random distractors per country per reference source; 5,000 test rows per country
per source. Useful for baseline training/debugging, not reliable leaderboard
estimation. Sampled test files cannot produce a full challenge submission.
"""

import argparse
import csv
import io
import json
import random
import sys
import tempfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path, PurePosixPath


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
SOURCE_FILES = [f"{split}_source{i}.tsv" for split in ("train", "test")
                for i in (1, 2, 3)]
EXTRAS = {"Documentation_template.md", "validate_submission.py"}


class InputFiles:
    def __init__(self, location):
        self.path = Path(location).expanduser()
        self.archive = None
        self.files = {}
        wanted = set(SOURCE_FILES) | {"train_ground_truth.tsv"} | EXTRAS
        if self.path.is_dir():
            entries = ((p.name, p) for p in self.path.rglob("*") if p.is_file())
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self.archive = zipfile.ZipFile(self.path)
            entries = ((PurePosixPath(i.filename).name, i)
                       for i in self.archive.infolist() if not i.is_dir()
                       and "__MACOSX" not in PurePosixPath(i.filename).parts)
        else:
            raise ValueError(f"Input must be an existing extracted folder or ZIP: {self.path}")
        for name, entry in entries:
            if name not in wanted:
                continue
            if name in self.files:
                raise ValueError(f"Multiple copies of {name} found. Choose a folder/ZIP with one dataset.")
            self.files[name] = entry
        missing = (set(SOURCE_FILES) | {"train_ground_truth.tsv"}) - self.files.keys()
        if missing:
            raise ValueError("Missing required files: " + ", ".join(sorted(missing)))

    @contextmanager
    def open(self, name):
        entry = self.files[name]
        raw = self.archive.open(entry) if self.archive else entry.open("rb")
        with raw:
            with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                yield text

    def size(self, name):
        entry = self.files[name]
        return entry.file_size if self.archive else entry.stat().st_size

    def close(self):
        if self.archive:
            self.archive.close()


def rows(source, name, required):
    with source.open(name) as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        absent = set(required) - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f"{name}: missing columns {sorted(absent)}; expected a TSV file.")
        for row in reader:
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f"{name}: malformed record near physical line {reader.line_num}.")
            yield {key: row[key] for key in required}


class Reservoir:
    """Bounded uniform random sampling within each country label."""
    def __init__(self, per_country, seed):
        self.limit = per_country
        self.random = random.Random(seed)
        self.seen = Counter()
        self.groups = {}

    def add(self, row):
        country = row["country"].strip() or "<missing>"
        self.seen[country] += 1
        bucket = self.groups.setdefault(country, [])
        if len(bucket) < self.limit:
            bucket.append(row)
        else:
            index = self.random.randrange(self.seen[country])
            if index < self.limit:
                bucket[index] = row

    def result(self):
        return [row for country in sorted(self.groups) for row in self.groups[country]]


class Profile:
    def __init__(self, size):
        self.count = 0
        self.size = size
        self.countries = Counter()
        self.missing = Counter()

    def add(self, row):
        self.count += 1
        self.countries[row["country"].strip() or "<missing>"] += 1
        for key in SOURCE_COLUMNS:
            if not row[key].strip():
                self.missing[key] += 1

    def result(self):
        return {"rows": self.count, "uncompressed_bytes": self.size,
                "country_counts": dict(sorted(self.countries.items())),
                "blank_field_counts": {key: self.missing[key] for key in SOURCE_COLUMNS}}


def write_tsv(archive, path, columns, data):
    with archive.open(path, "w") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)


def package(source, destination, args):
    summary = {"seed": args.seed, "full_dataset": {}, "sample": {},
               "purpose": "Baseline development; sampled-pool validation is optimistic and test is incomplete.",
               "sampling_config": {"train_per_country": args.train_per_country,
                   "distractors_per_country": args.distractors_per_country,
                   "test_per_country": args.test_per_country},
               "sample_country_counts": {}}
    name = "train_source1.tsv"
    print(f"Scanning {name} ...", flush=True)
    reservoir = Reservoir(args.train_per_country, args.seed)
    profile = Profile(source.size(name))
    for row in rows(source, name, SOURCE_COLUMNS):
        profile.add(row)
        reservoir.add(row)
    train_s1 = sorted(reservoir.result(), key=lambda r: r["entity_id"])
    selected = {r["entity_id"] for r in train_s1}
    if len(selected) != len(train_s1):
        raise ValueError("Duplicate IDs in the selected training Source 1 records.")
    summary["full_dataset"][name] = profile.result()
    summary["sample"][name] = len(train_s1)
    summary["sample_country_counts"][name] = dict(Counter(r["country"].strip() for r in train_s1))
    for country, count in summary["sample_country_counts"][name].items():
        if count < args.train_per_country:
            print(f"Note: {country} has only {count:,} available S1 records; kept all.", flush=True)

    print("Scanning train_ground_truth.tsv ...", flush=True)
    truth = {}
    required_ids = {"S2": set(), "S3": set()}
    match_histogram = Counter()
    truth_count = 0
    for row in rows(source, "train_ground_truth.tsv", TRUTH_COLUMNS):
        ids = [value.strip() for value in row["matched_entity_ids"].split(",") if value.strip()]
        match_histogram[len(ids)] += 1
        truth_count += 1
        sid = row["source1_entity_id"]
        if sid not in selected:
            continue
        if sid in truth:
            raise ValueError(f"Duplicate ground-truth row for sampled entity {sid}.")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate matching IDs in ground truth for {sid}.")
        for match in ids:
            prefix = match.split("-", 1)[0]
            if prefix not in required_ids:
                raise ValueError(f"Unexpected matching ID: {match}")
            required_ids[prefix].add(match)
        truth[sid] = {"source1_entity_id": sid, "matched_entity_ids": ",".join(ids)}
    absent_truth = selected - truth.keys()
    if absent_truth:
        raise ValueError(f"Ground truth is missing for {len(absent_truth)} sampled Source 1 records.")
    summary["full_dataset"]["train_ground_truth.tsv"] = {
        "rows": truth_count, "singleton_rows": match_histogram[0],
        "singleton_fraction": match_histogram[0] / truth_count if truth_count else None,
        "matches_per_entity_histogram": dict(sorted(match_histogram.items()))}
    summary["sample"]["train_ground_truth.tsv"] = len(truth)

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        write_tsv(out, "dataset/train/train_source1.tsv", SOURCE_COLUMNS, train_s1)
        write_tsv(out, "dataset/train/train_ground_truth.tsv", TRUTH_COLUMNS,
                  [truth[sid] for sid in sorted(truth)])

        for number in (2, 3):
            name = f"train_source{number}.tsv"
            print(f"Scanning {name}; keeping all selected businesses' matches ...", flush=True)
            wanted = required_ids[f"S{number}"]
            found = set()
            profile = Profile(source.size(name))
            extras = Reservoir(args.distractors_per_country, args.seed + number)
            kept_count = 0
            # Stream required matches into the ZIP; keep only distractor reservoirs in memory.
            with out.open(f"dataset/train/{name}", "w") as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=SOURCE_COLUMNS,
                                            delimiter="\t", lineterminator="\n")
                    writer.writeheader()
                    for row in rows(source, name, SOURCE_COLUMNS):
                        profile.add(row)
                        rid = row["entity_id"]
                        if rid in wanted:
                            if rid in found:
                                raise ValueError(f"Duplicate required record ID in {name}: {rid}")
                            found.add(rid)
                            writer.writerow(row)
                            kept_count += 1
                        else:
                            extras.add(row)
                    distractors = sorted(extras.result(), key=lambda r: r["entity_id"])
                    writer.writerows(distractors)
                    kept_count += len(distractors)
            missing = wanted - found
            if missing:
                raise ValueError(f"{name}: {len(missing)} matching IDs referenced by sampled labels are absent.")
            summary["full_dataset"][name] = profile.result()
            summary["sample"][name] = {"rows": kept_count, "required_matches": len(found),
                                      "random_distractors": len(distractors)}

        for number in (1, 2, 3):
            name = f"test_source{number}.tsv"
            print(f"Scanning {name} ...", flush=True)
            profile = Profile(source.size(name))
            reservoir = Reservoir(args.test_per_country, args.seed + 10 + number)
            for row in rows(source, name, SOURCE_COLUMNS):
                profile.add(row)
                reservoir.add(row)
            chosen = sorted(reservoir.result(), key=lambda r: r["entity_id"])
            write_tsv(out, f"dataset/test/{name}", SOURCE_COLUMNS, chosen)
            summary["full_dataset"][name] = profile.result()
            summary["sample"][name] = len(chosen)

        for name in sorted(EXTRAS & source.files.keys()):
            target = f"utils/{name}" if name.endswith(".py") else name
            with source.open(name) as stream:
                out.writestr(target, stream.read())
        out.writestr("dataset_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        out.writestr("SAMPLE_README.txt", (
            "BASELINE DEVELOPMENT SAMPLE - Amazon Business Entity Resolution\n\n"
            "Training: uniform random Source 1 sample within each country; ALL known S2/S3\n"
            "matches for selected S1 records are retained, plus random distractors.\n"
            "Countries are sampled separately, so the sample does not preserve country proportions.\n"
            "Test: independent random samples within each country and source. No test labels\n"
            "exist, so true test matches are NOT guaranteed to remain in this sample.\n"
            "Summary counts describe the FULL source files, not just the sample.\n"
            "Blank counts use empty/whitespace fields; literal NA/NULL strings are preserved.\n\n"
            "Use this ZIP for schema checks, examples, and developing the pipeline.\n"
            "You may train a quick baseline here, but validation is optimistic:\n"
            "many difficult distractors have been removed. Evaluate retrieval against the full pool.\n"
            "Sampled test outputs are not valid full challenge submissions.\n"
            "All original files were read only; this program performs no network requests.\n"))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Dataset ZIP, or extracted folder containing train/test TSV files")
    parser.add_argument("--output", default="er_sample.zip", help="New ZIP to create (default: er_sample.zip)")
    parser.add_argument("--train-per-country", type=int, default=5000, help="Records per country (default: 5000)")
    parser.add_argument("--distractors-per-country", type=int, default=5000, help="Records per country (default: 5000)")
    parser.add_argument("--test-per-country", type=int, default=5000, help="Test rows per country per source (default: 5000)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if min(args.train_per_country, args.distractors_per_country, args.test_per_country) < 1:
        parser.error("Sample sizes must be positive integers.")
    # csv's default field limit is too small for unusually long business descriptions.
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        parser.error(f"Output already exists: {destination}. Choose a different --output name.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = InputFiles(args.input)
    try:
        with tempfile.TemporaryDirectory(prefix="er_sample_", dir=destination.parent) as temporary:
            staging = Path(temporary) / "sample.zip"
            package(source, staging, args)
            # Refuse to overwrite even if a file appeared while the data was being scanned.
            with staging.open("rb") as src, destination.open("xb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
    finally:
        source.close()
    print(f"\nCreated: {destination}")
    print(f"ZIP size: {destination.stat().st_size / (1024 * 1024):.2f} MiB")
    print("Use this ZIP for baseline development. Keep the full test dataset for actual submission.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
