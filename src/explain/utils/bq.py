from google.cloud import bigquery


def retrieve_from_bigquery(sql: str, as_dataframe: bool = True, project: str = "datalake-prod-ef49c0c9"):
    """Retrieve data from datalake-prod-ef49c0c9 database
    Args:
        sql: SQL query to execute
        as_dataframe: Whether to return the results as a pandas dataframe
        project: BigQuery project ID
    """
    client = bigquery.Client(project=project)

    query_job = client.query(sql)
    if as_dataframe:
        results = query_job.to_dataframe()

    return results
