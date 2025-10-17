import os
import re
import json
import datetime
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from urllib.parse import urlparse, unquote
import yaml


# --------------------
# Helper functions
# --------------------
def ignore_email(text):
    if isinstance(text, str) and text.lower().startswith("received"):
        return str(text).split("\n", 1)[-1]
    return text

def t1_change(group):
    group_upper = str(group).upper()
    if "FIELDSUPPORT" in group_upper and "HDH" not in group_upper:
        return "FIELDSUPPORT-INTAKE"
    if "SECURITYZOOM" in group_upper:
        return "ZOOM"
    if "HDH" in group_upper:
        return group_upper.replace("HDH", "RRSS").replace("ITS-", "").replace("ITS", "")
    if "RMP" in group_upper:
        return "RMP"
    if "LIBRARY" in group_upper:
        return "LIBRARY"
    if "SECURITY" in group_upper:
        return "SECURITYSOC"
    if "FIS-" in group_upper:
        return "FIS"
    if "DATACOMM" in group_upper:
        return "DATACOMM"
    if "SERVICEDESK" in group_upper:
        return "SERVICEDESK"
    if "SNOW" in group_upper:
        return "SNOW"
    if "ONBASE" in group_upper:
        return "ONBASE"
    if "SIS-ACADEMICAFFAIRS" in group_upper:
        return "SIS-ACADEMICAFFAIRS"
    if "ATS-SIC" in group_upper:
        return "ETS-SIC"
    if "DATAMGMT" in group_upper:
        return "DATAINTEGRATIONSERVICES"
    if group_upper == "RRSS-COREBIO":
        return "RRSS"
    if group_upper == "JAMF":
        return "WORKSTATIONLIFECYCLE"
    if group_upper == "IAM-LASTPASS":
        return "OIA-IAM-LASTPASS"
    if group_upper == "BIA-VCHSDEVELOPMENT":
        return "BIA"
    return group_upper

def replace_links_and_emails(text):
    def extract_base_domain(url):
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"

    def unwrap_urldefense(url):
        match = re.search(r"__https?[^_<>]+", url)
        if match:
            return match.group(0)[2:]
        return url

    if not isinstance(text, str):
        return text

    # Remove mailto
    text = re.sub(r"\[mailto:[^\]]+\]", "", text)
    text = re.sub(r"<mailto:[^>]+>", "", text)

    # Replace URLs
    def link_replacer(match):
        url = match.group(0)
        if "urldefense.com" in url:
            url = unwrap_urldefense(url)
        url = unquote(url)
        base = extract_base_domain(url)
        return f"[LINK: {base}]"

    text = re.sub(r"<(https?://[^>\s]+)>", link_replacer, text)
    text = re.sub(r"https?://[^\s<>\"\)]+", link_replacer, text)

    # Replace emails
    def email_replacer(match):
        email = match.group(0)
        if "@" in email:
            _, domain = email.split("@", 1)
            return f"[EMAIL: {domain}]"
        return "[EMAIL]"

    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", email_replacer, text)
    return text.strip()

def remove_repeated_chars(text, max_repeats=5):
    if not isinstance(text, str):
        return text
    return re.sub(rf"(.)\1{{{max_repeats},}}", "", text)


# --------------------
# Main pipeline
# --------------------
def process_and_save(args, config):
    # Output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(args.output_dir, f"processed_{timestamp}")
    os.makedirs(folder, exist_ok=True)

    # Load data
    df = pd.read_csv(
        args.input_csv, encoding="utf-8", encoding_errors="ignore", engine="c", on_bad_lines="warn"
    )

    # Drop some unwanted groups
    df = df[df["assignment_group"] != "ITS-SecurityAbuse"]
    df = df[df["assignment_group"].notna()]
    df = df[df["assignment_group"].str.upper().str.startswith("ITS")]

    # Clean assignment_group
    df["assignment_group"] = (
        df["assignment_group"].str.upper().str.replace(r"^ITS[-]?", "", regex=True)
    )
    df = df.assign(
        assignment_group=df["assignment_group"].apply(t1_change),
        description=df["description"].apply(ignore_email),
    )

    # Valid groups
    defunct = set(config.get("defunct", []))
    valid_groups = (
        ~df["assignment_group"].isin(defunct)
        & (df["service_offering"].str.strip().str.lower() != "spam / duplicate case")
    )
    df = df[valid_groups].copy()

    # Combine text fields
    df = df[df["comments_and_work_notes"].notna()].copy()
    df["combined"] = (
        df["short_description"].fillna("") + " #### " + df["description"].fillna("")
    )
    df["combined"] = df["combined"].apply(replace_links_and_emails)
    df["combined"] = df["combined"].apply(
        lambda t: remove_repeated_chars(t, max_repeats=config.get("max_repeats", 5))
    )
    df["combined"] = df["combined"].str.replace("\n", " ")
    df["combined"] = df["combined"].str.replace("\r", " ")
    df["combined"] = df["combined"].fillna("")

    # Label encoding
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["assignment_group"])
    id2label = {int(i): label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: int(i) for i, label in enumerate(label_encoder.classes_)}

    # Save mappings
    with open(os.path.join(folder, "label_mappings.json"), "w") as f:
        json.dump(
            {"id2label": id2label, "label2id": label2id},
            f,
            indent=4
        )
    # Train/val split
    train_idx, val_idx = train_test_split(
        df.index,
        test_size=config.get("val_size", 0.1),
        random_state=config.get("random_state", 42),
        shuffle=True,
        stratify=df["label"],
    )
    small_df = df[["combined", "assignment_group", "label"]]

    train_df = small_df.loc[train_idx]
    val_df = small_df.loc[val_idx]

    # Save CSVs
    train_df.to_csv(os.path.join(folder, "train.csv"), index=False)
    val_df.to_csv(os.path.join(folder, "val.csv"), index=False)
    df.to_csv(os.path.join(folder, "all.csv"), index=False)

    print(f"Saved processed data to {folder}")