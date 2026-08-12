"""Clean ChEMBL E. coli ATCC 25922 MIC records for single-molecule modelling.

This script deliberately uses conservative exclusion rules: a questionable row is
removed instead of repaired or selected arbitrarily. It preserves all original
ChEMBL columns in the output so every retained record remains auditable.
"""

from pathlib import Path
import math

import pandas as pd
from rdkit import Chem


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path("dataprocess/chembl_ecoli_MIC_raw.csv")
OUTPUT_DIR = Path("dataprocess")


# Columns required to enforce the complete cleaning contract.
REQUIRED_COLUMNS = {
    "canonical_smiles",
    "standard_value",
    "standard_type",
    "standard_units",
    "standard_relation",
    "relation",
    "standard_flag",
    "data_validity_comment",
    "potential_duplicate",
    "target_pref_name",
    "target_organism",
    "target_tax_id",
    "assay_description",
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "assay_chembl_id",
    "document_chembl_id",
}


def report_filter(step: str, before: int, after: int) -> None:
    """Print the number and percentage of rows removed by one rule."""
    removed = before - after
    percentage = 100 * removed / before if before else 0.0
    print(f"\n{step}")
    print(f"Rows removed: {removed} ({percentage:.2f}%)")
    print(f"Rows retained: {after}")


def keep_rows(frame: pd.DataFrame, mask: pd.Series, step: str) -> pd.DataFrame:
    """Apply a Boolean filter and report its effect."""
    before = len(frame)
    result = frame.loc[mask].copy()
    report_filter(step, before, len(result))
    return result


def nonempty_text(series: pd.Series) -> pd.Series:
    """Return True for populated, non-whitespace text values."""
    return series.notna() & series.astype("string").str.strip().ne("")


# ============================================================
# Load data and validate schema
# ============================================================

df = pd.read_csv(INPUT_PATH, low_memory=False)

missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
if missing_columns:
    raise ValueError(f"Input CSV is missing required columns: {missing_columns}")

print("=" * 60)
print("Raw ChEMBL data")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Unique canonical SMILES: {df['canonical_smiles'].nunique(dropna=True)}")

clean = df.copy()


# ============================================================
# 1. Require complete identifiers, structure, assay provenance,
#    and MIC label fields. Empty strings count as missing.
# ============================================================

complete_columns = [
    "canonical_smiles",
    "standard_value",
    "standard_type",
    "standard_units",
    "standard_relation",
    "relation",
    "standard_flag",
    "target_pref_name",
    "target_organism",
    "target_tax_id",
    "assay_description",
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "assay_chembl_id",
    "document_chembl_id",
]

complete_mask = pd.Series(True, index=clean.index)
for column in complete_columns:
    complete_mask &= nonempty_text(clean[column])

clean = keep_rows(clean, complete_mask, "[1] Complete required fields")


# ============================================================
# 2. MIC must be numeric, finite, and strictly positive.
#
# MIC = 0 is physically invalid and log2(0) is undefined. NaN,
# infinity, negative values, and non-numeric strings are removed.
# ============================================================

clean["standard_value"] = pd.to_numeric(clean["standard_value"], errors="coerce")
mic_mask = (
    clean["standard_value"].notna()
    & clean["standard_value"].gt(0)
    & clean["standard_value"].lt(float("inf"))
)
clean = keep_rows(clean, mic_mask, "[2] MIC is finite and > 0")


# ============================================================
# 3. Keep one consistent endpoint, unit, and exact relation.
#
# Censored labels such as <, <=, >, and >= require bounded-loss
# handling and therefore are excluded from this exact-value baseline.
# Both ChEMBL relation fields must agree on '='.
# ============================================================

endpoint_mask = (
    clean["standard_type"].astype("string").str.strip().eq("MIC")
    & clean["standard_units"].astype("string").str.strip().eq("ug.mL-1")
    & clean["standard_relation"].astype("string").str.strip().eq("=")
    & clean["relation"].astype("string").str.strip().eq("=")
)
clean = keep_rows(
    clean,
    endpoint_mask,
    "[3] Endpoint = MIC; unit = ug.mL-1; relation = '='",
)


# ============================================================
# 4. Keep only standardized, valid, non-duplicate ChEMBL records.
#
# - standard_flag must be 1.
# - any data_validity_comment (including 'Outside typical range')
#   is treated as a warning and excluded conservatively.
# - potential_duplicate must be 0.
# ============================================================

standard_flag = pd.to_numeric(clean["standard_flag"], errors="coerce")
potential_duplicate = pd.to_numeric(clean["potential_duplicate"], errors="coerce")
validity_comment = clean["data_validity_comment"].fillna("").astype("string").str.strip()

quality_mask = standard_flag.eq(1) & validity_comment.eq("") & potential_duplicate.eq(0)
clean = keep_rows(
    clean,
    quality_mask,
    "[4] ChEMBL standard_flag = 1; no validity warning; not a potential duplicate",
)


# ============================================================
# 5. Keep only whole-cell Escherichia coli ATCC 25922.
#
# ChEMBL has no dedicated strain column in this export, so ATCC
# 25922 is verified from assay_description. Target name, organism,
# and taxonomy ID must also agree with E. coli.
# ============================================================

organism_mask = (
    clean["target_pref_name"].astype("string").str.strip().eq("Escherichia coli")
    & clean["target_organism"].astype("string").str.strip().eq("Escherichia coli")
    & clean["target_tax_id"].astype("string").str.strip().eq("562")
    & clean["assay_description"].str.contains(
        r"\bATCC\s*25922\b",
        case=False,
        regex=True,
        na=False,
    )
)
clean = keep_rows(clean, organism_mask, "[5] Organism/strain = E. coli ATCC 25922")


# ============================================================
# 6. Exclude synergy or combination experiments.
#
# A MIC measured in the presence of another compound is not a
# single-molecule label. 'synerg' matches synergy and synergistic.
# ============================================================

single_agent_assay_mask = ~clean["assay_description"].str.contains(
    r"synerg|combination|combined with|in the presence of|presence of",
    case=False,
    regex=True,
    na=False,
)
clean = keep_rows(clean, single_agent_assay_mask, "[6] Single-agent assay only")


# ============================================================
# 7. Keep only broth/microdilution MIC protocols.
#
# The source records span hundreds of assays and publications. An
# assay/document ID is provenance, not an error, so deleting IDs is
# not scientifically meaningful. Protocol heterogeneity is reduced by
# retaining only descriptions that explicitly identify broth dilution,
# microdilution, or the CLSI/NCCLS M07 broth-dilution standard.
#
# Agar/disk/diffusion/E-test/plate-dilution methods are explicitly
# excluded even if another accepted keyword is also present. Records
# with no stated method are excluded rather than guessed.
# ============================================================

description = clean["assay_description"].fillna("").astype("string")

accepted_mic_protocol = description.str.contains(
    r"micro[ -]?dilution"
    r"|microbroth\s+dilution"
    r"|micro\s+broth\s+dilution"
    r"|broth[- ]based\s+(?:micro)?dilution"
    r"|broth[- ]dilution"
    r"|broth\s+(?:micro|macro)?dilution"
    r"|\b(?:CLSI|NCCLS)\s+M0?7(?:\b|[- ])",
    case=False,
    regex=True,
    na=False,
)

excluded_nonbroth_protocol = description.str.contains(
    r"agar"
    r"|disc"
    r"|disk"
    r"|diffusion"
    r"|kirby[- ]bauer"
    r"|e[- ]?test"
    r"|cup[- ]?plate"
    r"|plate\s+dilution"
    r"|serial\s+plate",
    case=False,
    regex=True,
    na=False,
)

protocol_mask = accepted_mic_protocol & ~excluded_nonbroth_protocol
clean = keep_rows(
    clean,
    protocol_mask,
    "[7] Explicit broth/microdilution MIC protocol",
)


# ============================================================
# 8. Exclude every multi-fragment structure.
#
# A dot in SMILES separates disconnected components. These may be
# salts, solvates, mixtures, or multiple active compounds. Rather
# than guessing the active parent structure, all are excluded from
# this conservative single-molecule baseline.
# ============================================================

single_fragment_mask = ~clean["canonical_smiles"].str.contains(
    ".",
    regex=False,
    na=False,
)
clean = keep_rows(clean, single_fragment_mask, "[8] Single-fragment SMILES only")


# ============================================================
# 9. Require an RDKit-parseable molecular structure.
#
# Invalid structures cannot be featurized by Chemprop. Sanitization
# is enabled so chemically invalid valence/aromaticity is rejected.
# ============================================================

parseable_mask = clean["canonical_smiles"].map(
    lambda smiles: Chem.MolFromSmiles(str(smiles), sanitize=True) is not None
)
clean = keep_rows(clean, parseable_mask, "[9] RDKit-parseable SMILES")


# ============================================================
# 10. Remove ALL canonical-SMILES duplicate groups.
#
# If one canonical SMILES has multiple MIC records, every record in
# that group is removed. No arbitrary assay value is selected.
# ============================================================

smiles_counts = clean["canonical_smiles"].value_counts(dropna=False)
duplicated_smiles = smiles_counts[smiles_counts > 1].index
duplicate_smiles_mask = ~clean["canonical_smiles"].isin(duplicated_smiles)
clean = keep_rows(
    clean,
    duplicate_smiles_mask,
    f"[10] Unique canonical SMILES (duplicate groups: {len(duplicated_smiles)})",
)


# ============================================================
# 11. Remove ALL duplicated parent-molecule groups.
#
# Different ChEMBL molecule records can represent alternative forms
# of the same parent molecule. If a parent ID still occurs more than
# once, every member of that group is excluded, whether its MICs agree
# or disagree. This guarantees one retained label per chemical parent.
# ============================================================

parent_counts = clean["parent_molecule_chembl_id"].value_counts(dropna=False)
duplicated_parents = parent_counts[parent_counts > 1].index
unique_parent_mask = ~clean["parent_molecule_chembl_id"].isin(duplicated_parents)
clean = keep_rows(
    clean,
    unique_parent_mask,
    f"[11] Unique parent molecule (duplicate groups: {len(duplicated_parents)})",
)


# ============================================================
# 12. Add the modelling target and verify final invariants.
#
# MIC is measured on a multiplicative dilution scale. log2(MIC)
# converts a two-fold MIC difference into one target unit.
# ============================================================

clean["log2_mic"] = clean["standard_value"].map(math.log2)

if clean.empty:
    raise RuntimeError("Cleaning removed every row; no modelling data remain.")

if clean["canonical_smiles"].duplicated().any():
    raise RuntimeError("Duplicate canonical SMILES remain after cleaning.")

if clean["parent_molecule_chembl_id"].duplicated().any():
    raise RuntimeError("Duplicate parent molecule IDs remain after cleaning.")

if not clean["standard_value"].gt(0).all():
    raise RuntimeError("Non-positive MIC values remain after cleaning.")

if clean["canonical_smiles"].str.contains(".", regex=False).any():
    raise RuntimeError("Multi-fragment SMILES remain after cleaning.")

if clean["data_validity_comment"].fillna("").astype("string").str.strip().ne("").any():
    raise RuntimeError("ChEMBL validity warnings remain after cleaning.")

if pd.to_numeric(clean["standard_flag"], errors="coerce").ne(1).any():
    raise RuntimeError("Non-standard ChEMBL records remain after cleaning.")

if clean["assay_description"].str.contains(
    r"synerg|combination|combined with|in the presence of|presence of",
    case=False,
    regex=True,
    na=False,
).any():
    raise RuntimeError("Combination/synergy assays remain after cleaning.")

final_description = clean["assay_description"].fillna("").astype("string")
final_accepted_protocol = final_description.str.contains(
    r"micro[ -]?dilution"
    r"|microbroth\s+dilution"
    r"|micro\s+broth\s+dilution"
    r"|broth[- ]based\s+(?:micro)?dilution"
    r"|broth[- ]dilution"
    r"|broth\s+(?:micro|macro)?dilution"
    r"|\b(?:CLSI|NCCLS)\s+M0?7(?:\b|[- ])",
    case=False,
    regex=True,
    na=False,
)
if not final_accepted_protocol.all():
    raise RuntimeError("Records without an explicit broth/microdilution protocol remain.")

if not clean["log2_mic"].map(math.isfinite).all():
    raise RuntimeError("Non-finite log2(MIC) targets remain after cleaning.")


# ============================================================
# 13. Save (only when this script is run explicitly)
# ============================================================

n_rows = len(clean)
output_path = OUTPUT_DIR / f"chembl_ecoli_MIC_clean_{n_rows}.csv"
clean.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)
print(f"Final rows: {len(clean)}")
print(f"Unique canonical SMILES: {clean['canonical_smiles'].nunique()}")
print(f"Unique parent molecules: {clean['parent_molecule_chembl_id'].nunique()}")
print(
    "MIC range: "
    f"{clean['standard_value'].min()} - {clean['standard_value'].max()} ug/mL"
)
print(f"Saved to: {output_path}")
