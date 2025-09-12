import argparse
from pathlib import Path

import polars as pl
from google.cloud import bigquery


def main():
    parser = argparse.ArgumentParser(description="Fetch KCE rows from BigQuery and save to Parquet.")
    parser.add_argument("--project", default="datalake-dev-ea785c15", help="GCP project ID")
    parser.add_argument("--table", default="about_a_molecule.kce_details_vw", help="BigQuery dataset.table")
    parser.add_argument("--organism", default="Homo sapiens", help="Organism filter (exact match)")
    parser.add_argument("--out", type=Path, required=True, help="Output parquet path")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)

    fq_table = f"{args.project}.{args.table}"  # project.dataset.table

    # 1) Row count
    count_sql = f"SELECT COUNT(*) AS n FROM `{fq_table}`"
    n = list(client.query(count_sql).result())[0]["n"]
    print(f"[info] Row count in `{fq_table}`: {n}")

    # 2) Fetch rows for the organism (parameterized)
    sql = f"""
    SELECT
      inchi_key,
      rec_id,
      smiles,
      bioref_organism,
      bioref_original_measurement_value,
      bioref_original_measurement_unit,
      bioref_measurement_type,
      bioref_protein,
      bioref_protein_name,
      bioref_gene_name,
      bioref_protein_accession_id,
      bioref_original_id,
      bioref_article_doi,
      bioref_assay_type,
      mce_biological_activity,
      mce_target,
      mce_research_area,
      mce_pathway,
      mce_product_name,
      general_kce
    FROM `{fq_table}`
    WHERE bioref_organism = @organism
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("organism", "STRING", args.organism)]
    )

    job = client.query(sql, job_config=job_config)
    result = job.result()  # blocks
    df_all = pl.from_arrow(result.to_arrow())

    print(f"[info] Retrieved {df_all.height} rows for organism = '{args.organism}'")

    # Ensure output dir exists and write parquet
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_all.write_parquet(str(args.out))


if __name__ == "__main__":
    main()
