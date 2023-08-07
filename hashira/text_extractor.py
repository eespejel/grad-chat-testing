import json
import os
import re

from termcolor import cprint
from utils import load_config


def preprocess_text(text):
    cleaned_text = re.sub(r"\s+", " ", text)  # Removing extra white spaces
    cleaned_text = re.sub(r"\n", "", cleaned_text)  # Removing line breaks
    cleaned_text = re.sub(r"\f", "", cleaned_text)  # Removing page break markers
    cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text)  # Removing special characters
    cleaned_text = cleaned_text.strip()  # Removing leading and trailing spaces
    return cleaned_text


def load_and_preprocess_data():
    config = load_config()
    json_database_path = config.get("json_database_path", "")
    if not json_database_path:
        cprint("Database path is not found in configuration.", "red")
        return
    try:
        with open(json_database_path, "r") as file:
            data = json.load(file)

        cprint(f"Loaded data from {json_database_path}", "green")

        # Preprocessing "texto" in each item in the list of dicts
        for item in data:
            if "texto" in item:
                item["texto"] = preprocess_text(item["texto"])

            # Extract the date from "fecha_evento" and "fecha_adicion" keys
            for key in ["fecha_evento", "fecha_adicion"]:
                if key in item and isinstance(item[key], dict):
                    item[key] = item[key].get(
                        "$date", ""
                    )  # Replace with the value of $date, if exists.

        cprint(f"Finished preprocessing data", "green")

        # Saving the preprocessed data in a new JSONL file
        base_name = os.path.basename(json_database_path)
        base_name_without_ext = os.path.splitext(base_name)[0]
        preprocessed_file_name = base_name_without_ext + "_preprocessed.jsonl"
        preprocessed_file_path = os.path.join(
            os.path.dirname(json_database_path), preprocessed_file_name
        )
        with open(preprocessed_file_path, "w") as preprocessed_file:
            for item in data:
                preprocessed_file.write(json.dumps(item) + "\n")

        cprint(f"Saved preprocessed data to {preprocessed_file_path}", "green")
    except FileNotFoundError:
        cprint(f"No file found at {json_database_path}", "red")


load_and_preprocess_data()
