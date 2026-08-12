import pandas as pd

PATH = "dataprocess/chembl_ecoli_MIC_raw.csv"

df = pd.read_csv(PATH, low_memory=False)

print("=" * 60)
print("BASIC")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("Unique SMILES:", df["canonical_smiles"].nunique())
print("Unique molecule IDs:", df["molecule_chembl_id"].nunique())
print("Unique parent molecules:", df["parent_molecule_chembl_id"].nunique())
print("Unique assays:", df["assay_chembl_id"].nunique())
print("Unique documents:", df["document_chembl_id"].nunique())


# -------------------------------------------------
# 1. 确认是不是全部都是 E. coli + MIC
# -------------------------------------------------

print("\n" + "=" * 60)
print("TARGET")
print("=" * 60)

print("\nstandard_type:")
print(df["standard_type"].value_counts(dropna=False))

print("\ntarget_organism:")
print(df["target_organism"].value_counts(dropna=False).head(20))

print("\ntarget_pref_name:")
print(df["target_pref_name"].value_counts(dropna=False).head(20))


# -------------------------------------------------
# 2. MIC relationship
# -------------------------------------------------

print("\n" + "=" * 60)
print("RELATION")
print("=" * 60)

print(df["standard_relation"].value_counts(dropna=False))


# -------------------------------------------------
# 3. MIC units
# -------------------------------------------------

print("\n" + "=" * 60)
print("UNITS")
print("=" * 60)

print(df["standard_units"].value_counts(dropna=False).head(30))


# relation × unit
print("\nRelation × unit:")
print(
    pd.crosstab(
        df["standard_relation"],
        df["standard_units"],
        margins=True
    )
)


# -------------------------------------------------
# 4. 数值是否完整
# -------------------------------------------------

print("\n" + "=" * 60)
print("VALUE QUALITY")
print("=" * 60)

numeric_value = pd.to_numeric(
    df["standard_value"],
    errors="coerce"
)

print("Missing standard_value:", numeric_value.isna().sum())
print("Non-positive values:", (numeric_value <= 0).sum())

print("\nstandard_value describe:")
print(numeric_value.describe())


# -------------------------------------------------
# 5. SMILES
# -------------------------------------------------

print("\n" + "=" * 60)
print("SMILES")
print("=" * 60)

print("Missing SMILES:", df["canonical_smiles"].isna().sum())
print("Unique SMILES:", df["canonical_smiles"].nunique())

smiles_counts = df["canonical_smiles"].value_counts()

print("SMILES occurring >1 time:", (smiles_counts > 1).sum())
print("Max records for one SMILES:", smiles_counts.max())

print("\nTop repeated SMILES:")
print(smiles_counts.head(20))


# -------------------------------------------------
# 6. ChEMBL quality flags
# -------------------------------------------------

print("\n" + "=" * 60)
print("QUALITY FLAGS")
print("=" * 60)

print("\nstandard_flag:")
print(df["standard_flag"].value_counts(dropna=False))

print("\npotential_duplicate:")
print(df["potential_duplicate"].value_counts(dropna=False))

print("\ndata_validity_comment:")
print(df["data_validity_comment"].value_counts(dropna=False).head(30))


# -------------------------------------------------
# 7. assay
# -------------------------------------------------

print("\n" + "=" * 60)
print("ASSAYS")
print("=" * 60)

print("Unique assays:", df["assay_chembl_id"].nunique())

print("\nTop assays:")
print(df["assay_chembl_id"].value_counts().head(30))


# -------------------------------------------------
# 8. assay description
# -------------------------------------------------

print("\n" + "=" * 60)
print("ASSAY DESCRIPTIONS")
print("=" * 60)

print(
    df["assay_description"]
    .value_counts(dropna=False)
    .head(30)
)


# -------------------------------------------------
# 9. 文献年份
# -------------------------------------------------

print("\n" + "=" * 60)
print("DOCUMENT YEAR")
print("=" * 60)

print(df["document_year"].describe())

print("\nTop years:")
print(
    df["document_year"]
    .value_counts()
    .sort_index()
    .tail(30)
)