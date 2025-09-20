#!/usr/bin/env python3
import os, sys, json, argparse, pathlib
from typing import Any, Dict
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def load_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def validate_file(file_path: str, q_schema: Dict[str, Any], l_schema: Dict[str, Any]) -> int:
	data = load_json(file_path)
	errs = 0
	try:
		Draft202012Validator(q_schema).validate(data)
	except ValidationError as e:
		print(f"[ERROR] {os.path.basename(file_path)} quantitative: {e.message}", file=sys.stderr)
		errs += 1

	qual_obj = {"qualitative_thematic": data.get("qualitative_thematic", {"codes": []})}
	try:
		Draft202012Validator(l_schema).validate(qual_obj)
	except ValidationError as e:
		print(f"[ERROR] {os.path.basename(file_path)} qualitative: {e.message}", file=sys.stderr)
		errs += 1

	return errs


def main():
	ap = argparse.ArgumentParser(description="Validate extracted JSON files against schemas.")
	ap.add_argument("--in-dir", required=True, help="Directory with *.extracted.json files")
	ap.add_argument("--quant-schema", required=True, help="Path to data_schemas/quantitative_schema.json")
	ap.add_argument("--qual-schema", required=True, help="Path to data_schemas/qualitative_schema.json")
	args = ap.parse_args()

	q_schema = load_json(args.quant_schema)
	l_schema = load_json(args.qual_schema)

	in_dir = pathlib.Path(args.in_dir)
	files = sorted([p for p in in_dir.glob("*.extracted.json")])
	if not files:
		print(f"No extracted JSONs found in {in_dir}", file=sys.stderr)
		sys.exit(1)

	total_errs = 0
	for p in files:
		total_errs += validate_file(str(p), q_schema, l_schema)

	if total_errs:
		print(f"Validation finished with {total_errs} errors.")
		sys.exit(2)
	else:
		print("All files valid.")


if __name__ == "__main__":
	main()

