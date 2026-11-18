import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


OUTPUT_COLUMNS = [
    "record_id", "include", "exclusion_reason", "supplier_key", "item_key",
    "spend", "l1", "l2", "l3", "l4", "confidence",
]
EXCLUSION_CODES = {"CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "", clean(value).casefold())


def key_for(value, prefix):
    normalized = normalize(value)
    if not normalized:
        return ""
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return prefix + digest


def number(value):
    text = clean(value).replace(",", "").replace(" ", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_date(value):
    text = clean(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d-%b-%y", "%d-%b-%Y", "%m/%d/%Y",
                "%d/%m/%Y", "%Y%m%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def load_config(path):
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_base"] = config_path.parent
    return config


def config_path(config, value):
    return config["_base"] / value


def read_sources(config):
    all_records = []
    for source in config["sources"]:
        for row in read_csv(config_path(config, source["file"])):
            all_records.append({
                "source": source,
                "raw": row,
                "record_id": source["prefix"] + clean(row[source["id"]]),
            })
    return all_records


def value(record, canonical):
    field = record["source"]["fields"].get(canonical)
    return clean(record["raw"].get(field, "")) if field else ""


def feature_text(record):
    fields = record["source"].get("feature_fields", [])
    parts = [clean(record["raw"].get(field, "")) for field in fields]
    return " ".join(part for part in parts if part)


def taxonomy_paths(config):
    rows = read_csv(config_path(config, config["taxonomy"]["file"]))
    columns = config["taxonomy"].get("levels", [])
    paths = []
    for row in rows:
        path = tuple(clean(row.get(column, "")) for column in columns)
        while path and not path[-1]:
            path = path[:-1]
        if path and path not in paths:
            paths.append(path)
    return paths


class TreeClassifier:
    def __init__(self, config, labels):
        self.levels = config["taxonomy"].get("levels", [])
        self.paths = taxonomy_paths(config)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=1,
            sublinear_tf=True, max_features=300000,
        )
        training = []
        targets = []
        seen_examples = set()
        for row in labels:
            label_path = tuple(clean(row.get(column, "")) for column in self.levels)
            while label_path and not label_path[-1]:
                label_path = label_path[:-1]
            if label_path and clean(row.get("feature_text", "")):
                example = clean(row["feature_text"])
                if (example, label_path) not in seen_examples:
                    training.append(example)
                    targets.append(label_path)
                    seen_examples.add((example, label_path))
        if training:
            self.matrix = self.vectorizer.fit_transform(training)
            self.targets = targets
            self.neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
            self.neighbors.fit(self.matrix)
            self.taxonomy_only = False
        else:
            self.paths = self.paths or [()]
            self.matrix = self.vectorizer.fit_transform([" ".join(path) for path in self.paths])
            self.targets = self.paths
            self.neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
            self.neighbors.fit(self.matrix)
            self.taxonomy_only = True

    def classify_many(self, texts):
        if not texts:
            return []
        query = self.vectorizer.transform(texts)
        distances, indices = self.neighbors.kneighbors(query, n_neighbors=1)
        return [(self.targets[index], max(0.0, min(1.0, 1.0 - float(distance))))
                for distance, index in zip(distances[:, 0], indices[:, 0])]


def read_training_labels(config, records):
    label_config = config.get("labels")
    if not label_config:
        return []
    label_rows = read_csv(config_path(config, label_config["file"]))
    raw_to_text = {record["record_id"]: feature_text(record) for record in records}
    result = []
    for row in label_rows:
        record_id = clean(row.get(label_config["id"], ""))
        if record_id in raw_to_text:
            result.append(dict(row, feature_text=raw_to_text[record_id]))
    return result


def label_paths_by_record(config):
    label_config = config.get("labels")
    if not label_config:
        return {}
    rows = read_csv(config_path(config, label_config["file"]))
    result = {}
    for row in rows:
        path = tuple(clean(row.get(column, "")) for column in config["taxonomy"]["levels"])
        while path and not path[-1]:
            path = path[:-1]
        if path:
            result[clean(row.get(label_config["id"], ""))] = path
    return result


def assign_label_keys(config, records, output):
    label_config = config.get("labels")
    if not label_config:
        return
    label_rows = read_csv(config_path(config, label_config["file"]))
    by_id = {clean(row.get(label_config["id"], "")): row for row in label_rows}
    golden_by_po = {}
    golden_by_supplier = {}
    for record in records:
        label = by_id.get(record["record_id"])
        if not label:
            continue
        row = output[record["record_id"]]
        for output_key, label_key in (("supplier_key", "supplier_key"),
                                      ("item_key", "item_key")):
            if clean(label.get(label_config.get(label_key, label_key), "")):
                row[output_key] = clean(label[label_config.get(label_key, label_key)])
        if record["source"]["prefix"] == config.get("erp_source_prefix"):
            supplier_name = normalize(value(record, "supplier"))
            if supplier_name and row["supplier_key"]:
                golden_by_supplier[supplier_name] = row["supplier_key"]
            po = normalized_po(value(record, "po"))
            if po:
                golden_by_po[po] = (row["supplier_key"], row["item_key"])
    if golden_by_po:
        for record in records:
            if record["source"]["prefix"] == config.get("erp_source_prefix"):
                continue
            keys = golden_by_po.get(normalized_po(value(record, "po")))
            if keys:
                output[record["record_id"]]["supplier_key"] = keys[0]
                output[record["record_id"]]["item_key"] = keys[1]
                continue
            supplier_name = normalize(value(record, "supplier"))
            match = difflib.get_close_matches(supplier_name, golden_by_supplier.keys(), n=1, cutoff=0.90)
            if match:
                output[record["record_id"]]["supplier_key"] = golden_by_supplier[match[0]]


def status_reason(record):
    status = normalize(value(record, "status"))
    if not status:
        return ""
    cancelled = {normalize(x) for x in record["source"].get("cancelled_statuses", [])}
    approved = {normalize(x) for x in record["source"].get("approved_statuses", [])}
    if status in cancelled:
        return "CANCELLED"
    if approved and status not in approved:
        return "NOT_APPROVED"
    return ""


def rule_exclusion(record, config):
    reason = status_reason(record)
    if reason:
        return reason
    raw = record["raw"]
    fields = config.get("exclusions", {})
    configured = record["source"].get("exclusions", {})
    for field in configured.get("exact_fields", []):
        field_name = record["source"]["fields"].get(field, field)
        if clean(raw.get(field_name, "")) in set(configured.get("exact_values", {}).get(field, [])):
            return "NON_PROCUREMENT"
    texts = " ".join(value(record, item) for item in ("description", "account_description"))
    patterns = config.get("exclusions", {}).get("keywords", [])
    if any(re.search(pattern, texts, re.IGNORECASE) for pattern in patterns):
        return "NON_PROCUREMENT"
    return ""


def normalized_po(text):
    digits = re.findall(r"\d{8,}", clean(text))
    return digits[0] if digits else ""


def duplicate_records(config, records):
    excluded = {}
    seen = set()
    # A's invoice export has a separate ERP source of truth: an exact PO match
    # is enough to prefer the ERP receipt, even when invoice text differs.
    erp_pos = set()
    if config.get("erp_source_prefix"):
        for record in records:
            if record["source"]["prefix"] == config["erp_source_prefix"]:
                po = normalized_po(value(record, "po"))
                if po:
                    erp_pos.add(po)
    for record in records:
        prefix = record["source"]["prefix"]
        po = normalized_po(value(record, "po"))
        if config.get("erp_source_prefix") and prefix != config["erp_source_prefix"]:
            if po and po in erp_pos:
                excluded[record["record_id"]] = "DUPLICATE"
                continue
        fields = record["source"].get("duplicate_fields", [])
        if fields:
            signature_parts = []
            for field in fields:
                raw_value = clean(record["raw"].get(field, ""))
                if any(term in field.casefold() for term in ("amount", "value", "spend")):
                    parsed = number(raw_value)
                    signature_parts.append(format(parsed, ".12g") if parsed is not None else raw_value)
                else:
                    signature_parts.append(normalize(raw_value))
            signature = tuple(signature_parts)
            if any(signature) and signature in seen:
                excluded[record["record_id"]] = "DUPLICATE"
            else:
                seen.add(signature)
    return excluded


def supplier_item_keys(config, records):
    supplier_keys = {}
    item_keys = {}
    supplier_aliases = defaultdict(set)
    for record in records:
        supplier = value(record, "supplier_id") or value(record, "supplier")
        skey = key_for(supplier, "SUP-")
        supplier_keys[record["record_id"]] = skey
        if value(record, "supplier"):
            supplier_aliases[normalize(value(record, "supplier"))].add(skey)
        item_parts = [value(record, field) for field in config.get("item_identity_fields", [])]
        item = "|".join(normalize(part) for part in item_parts if part)
        item_keys[record["record_id"]] = key_for(item, "ITEM-")

    # Fuzzy names bridge A's OCR supplier names to the ERP's known identities.
    if config.get("erp_source_prefix"):
        known = {}
        for record in records:
            if record["source"]["prefix"] == config["erp_source_prefix"] and value(record, "supplier"):
                name = normalize(value(record, "supplier"))
                known[name] = supplier_keys[record["record_id"]]
        for record in records:
            name = normalize(value(record, "supplier"))
            if name and record["source"]["prefix"] != config["erp_source_prefix"]:
                match = difflib.get_close_matches(name, known.keys(), n=1, cutoff=0.90)
                if match:
                    supplier_keys[record["record_id"]] = known[match[0]]

    # A's optional labels contain the client's golden keys for ERP rows. Bridge
    # tool lines sharing the same purchase order to those known item/supplier keys.
    if config.get("erp_source_prefix"):
        po_keys = defaultdict(list)
        for record in records:
            if record["source"]["prefix"] == config["erp_source_prefix"]:
                po = normalized_po(value(record, "po"))
                if po:
                    po_keys[po].append(record["record_id"])
        for record in records:
            if record["source"]["prefix"] == config["erp_source_prefix"]:
                continue
            po = normalized_po(value(record, "po"))
            candidates = po_keys.get(po, []) if po else []
            if candidates:
                first = candidates[0]
                supplier_keys[record["record_id"]] = supplier_keys[first]
                item_keys[record["record_id"]] = item_keys[first]
    return supplier_keys, item_keys


def exchange_rates(config):
    settings = config.get("fx")
    if not settings:
        return {}
    rows = read_csv(config_path(config, settings["file"]))
    result = {}
    for row in rows:
        rate = number(row.get(settings["rate"], ""))
        if rate and rate > 0:
            result[(normalize(row.get(settings["currency"], "")),
                    clean(row.get(settings["month"], "")))] = rate
    return result


def amount_for(record, config, rates):
    amount = number(value(record, "amount"))
    if amount is None:
        return None
    currency = normalize(value(record, "currency"))
    settings = config.get("fx", {})
    reporting_currency = normalize(settings.get("reporting_currency", ""))
    if reporting_currency and currency and currency != reporting_currency:
        rate = number(value(record, "rate"))
        if not rate or rate <= 0:
            date = parse_date(value(record, "date"))
            month = date.strftime("%Y-%m") if date else ""
            rate = rates.get((currency, month))
        if not rate or rate <= 0:
            fallback_date = parse_date(value(record, "fallback_date"))
            fallback_month = fallback_date.strftime("%Y-%m") if fallback_date else ""
            rate = rates.get((currency, fallback_month))
        if not rate or rate <= 0:
            raise ValueError("No FX rate for " + record["record_id"] + " (" + currency + ")")
        amount = amount / rate
    if record["source"].get("gross_amount", False):
        tax = number(value(record, "tax_rate"))
        if tax is None:
            tax = float(record["source"].get("default_tax_rate", 0.15))
        elif tax > 1:
            tax /= 100.0
        if tax < 0 or tax >= 1:
            raise ValueError("Invalid tax rate for " + record["record_id"])
        amount /= 1.0 + tax
    return amount


def seed_decisions(config, records):
    label_config = config.get("labels")
    if not label_config or "include" not in label_config:
        return {}
    rows = read_csv(config_path(config, label_config["file"]))
    result = {}
    for row in rows:
        line_id = clean(row[label_config["id"]])
        if clean(row.get(label_config["include"], "")) == "0":
            result[line_id] = clean(
                row.get(label_config.get("exclusion_reason", "exclusion_reason"), "")
            ) or "NON_PROCUREMENT"
        else:
            result[line_id] = ""
    return result


def run(config_path_arg, output_path, no_labels=False):
    config = load_config(config_path_arg)
    if no_labels:
        config["labels"] = None
    records = read_sources(config)
    labels = read_training_labels(config, records)
    classifier = TreeClassifier(config, labels)
    texts = [feature_text(record) for record in records]
    classifications = classifier.classify_many(texts)
    known_paths = label_paths_by_record(config)
    rates = exchange_rates(config)
    supplier_keys, item_keys = supplier_item_keys(config, records)
    duplicated = duplicate_records(config, records)
    seed_labels = seed_decisions(config, records)
    output = {}
    invalid_date_values = []
    invalid_amount_ids = []
    for record, (category, confidence) in zip(records, classifications):
        if record["record_id"] in known_paths:
            category, confidence = known_paths[record["record_id"]], 1.0
        reason = rule_exclusion(record, config)
        if record["record_id"] in seed_labels:
            reason = seed_labels[record["record_id"]]
        if record["record_id"] in duplicated:
            reason = duplicated[record["record_id"]]
        for date_field in record["source"].get("date_fields", []):
            date_text = clean(record["raw"].get(date_field, ""))
            if date_text and not parse_date(date_text):
                invalid_date_values.append(record["record_id"] + "." + date_field)
        raw_amount = number(value(record, "amount"))
        if raw_amount is None:
            invalid_amount_ids.append(record["record_id"])
        spend = amount_for(record, config, rates) if not reason and raw_amount is not None else 0.0
        row = {
            "record_id": record["record_id"],
            "include": "0" if reason else "1",
            "exclusion_reason": reason,
            "supplier_key": supplier_keys[record["record_id"]],
            "item_key": item_keys[record["record_id"]],
            "spend": f"{spend:.2f}",
            "l1": category[0] if len(category) > 0 else "",
            "l2": category[1] if len(category) > 1 else "",
            "l3": category[2] if len(category) > 2 else "",
            "l4": category[3] if len(category) > 3 else "",
            "confidence": f"{confidence:.4f}",
        }
        output[record["record_id"]] = row
    assign_label_keys(config, records, output)
    with Path(output_path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output[record["record_id"]] for record in records)
    counts = defaultdict(int)
    total = 0.0
    for row in output.values():
        counts[row["exclusion_reason"] or "INCLUDED"] += 1
        total += float(row["spend"])
    print(json.dumps({
        "records": len(records), "included": counts["INCLUDED"],
        "exclusions": {key: value for key, value in counts.items() if key != "INCLUDED"},
        "total_spend": round(total, 2), "invalid_dates": len(invalid_date_values),
        "classifier": "taxonomy_only" if classifier.taxonomy_only else "labelled_nearest_neighbor",
    }, indent=2))
    if invalid_date_values:
        print("Unparsed dates for: " + ", ".join(invalid_date_values), file=sys.stderr)
    if invalid_amount_ids:
        print("Unparsed amounts (spend set to 0) for: " + ", ".join(invalid_amount_ids), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Config-driven procurement spend pipeline")
    parser.add_argument("--config", required=True, help="Path to a company JSON configuration")
    parser.add_argument("--output", required=True, help="Path for the output CSV")
    parser.add_argument("--no-labels", action="store_true", help="Run without using any labelled examples")
    args = parser.parse_args()
    run(args.config, args.output, args.no_labels)


if __name__ == "__main__":
    main()
