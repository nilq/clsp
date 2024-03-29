"""Storage utilities."""

from google.cloud import storage, bigquery
from google.api_core.retry import Retry
from functools import cache

import polars as pl
import io
from urllib.parse import urlparse

from typing import Optional


@cache
def gcs_client(project: Optional[str] = None) -> storage.Client:
    """Get cached GCS client for project.

    Args:
        project (str): GCP project, otherwise default project.
            Defaults to None

    Returns:
        storage.Client: Cloud Storage client.
    """
    return storage.Client(project=project)


@cache
def bq_client(project: Optional[str] = None) -> bigquery.Client:
    """Get cached BigQuery client for project.

    Args:
        project (str): GCP project, otherwise default project.
            Defaults to None

    Returns:
        bigquery.Client: BigQuery client.
    """
    return bigquery.Client(project=project)


def blob_from_url(url: str, project: Optional[str] = None) -> storage.Blob:
    """Get Blob from URL.

    Args:
        url (str): URL of Blob.
        project (str): Project to get Blob with.

    Returns:
        storage.Blob: The Blob you wanted.
    """
    client = gcs_client(project)
    parts = urlparse(url)

    return client.bucket(parts.netloc).blob(parts.path.lstrip("/"))


def gcs_download_bytes(
    url: str,
    raw_download: bool = False,
    do_retry: bool = False,
    project: Optional[str] = None,
) -> io.BytesIO:
    """Download bytes of blob.

    Args:
        url (str): GCS URL to download from.
        do_retry (bool, optional): Whether to retry with retry defaults.
            Defaults to False.
        raw_download (bool, optional): Whether to download bytes without expansion.
        project (Optional[str]): Project, otherwise will use default.
            Defaults to None.

    Returns:
        io.BytesIO: Buffer with bytes.
    """
    blob = blob_from_url(url)
    retry = Retry() if do_retry else None
    return blob.download_as_bytes(raw_download=raw_download, retry=retry)


def bigquery_query(query: str, project: Optional[str]) -> pl.DataFrame:
    """Run BigQuery query, return result.

    Args:
        query (str): Query to run.
        project (Optional[str]): Project, otherwise will use default.
            Defaults to None.

    Returns:
        pl.DataFrame: DataFrame containing result of query.
    """
    client = bq_client(project=project)
    return pl.from_arrow(client.query(query).result().to_arrow())
