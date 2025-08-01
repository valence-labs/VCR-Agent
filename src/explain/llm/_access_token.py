import os

import dotenv
from google.cloud import secretmanager

dotenv.load_dotenv()

# set the Google Cloud project if env is set, otherwise default to rxrx-medchem-auto-dev
google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT", "rxrx-medchem-auto-dev")


def access_secret_version(
    secret_name: str,
    project_id: str = google_cloud_project,
    secret_ver: int | str = "latest",
) -> str:
    """
    Accesses a secret version from Google Cloud Secret Manager.

    Args:
        secret_name (str): The name of the secret.
        project_id (str, optional): The project ID. Defaults to GOOGLE_CLOUD_PROJECT env var or "rxrx-medchem-auto-dev".

        secret_ver (Union[int, str], optional): The version of the secret to access. Defaults to "latest".

    Returns:
        str: The secret data.

    Examples:
        # Access the latest version of the secret
        >>> access_secret_version("my_secret")
        'my_secret_data'

        # Access a specific version of the secret
        >>> access_secret_version("my_secret", secret_ver=2)
        'my_secret_data_v2'
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/{secret_ver}"
    response = client.access_secret_version(request={"name": name})
    data = response.payload.data.decode("UTF-8")
    assert data is not None, "Secret not found"
    return data


def set_env_secrets(secret_list: list[str] | None = None):
    """
    Set environment variables from Google Cloud Secret Manager.

    Args:
        secret_list (Optional[list[str]], optional): The list of secrets to set. Defaults to ["OPENAI_API_KEY"].
    """
    if secret_list is None:
        secret_list = ["OPENAI_API_KEY"]
    for secret in secret_list:
        secret_data = access_secret_version(secret)
        os.environ[secret] = secret_data
