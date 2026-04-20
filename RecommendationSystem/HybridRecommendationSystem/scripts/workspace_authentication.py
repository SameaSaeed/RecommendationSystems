import os
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient

def get_workspace_client():
    """
    Returns a properly authenticated Databricks WorkspaceClient.
    Works locally using .env or inside Databricks workspace.
    """
    # Check if running inside Databricks
    running_in_db = "DATABRICKS_RUNTIME_VERSION" in os.environ

    if not running_in_db:
        # Load environment variables locally from .env
        load_dotenv()
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not databricks_host or not databricks_token:
            raise ValueError(
                "Databricks host or token not set. Please set DATABRICKS_HOST and DATABRICKS_TOKEN in .env"
            )

        # Set env vars so SDK picks them up
        os.environ["DATABRICKS_HOST"] = databricks_host
        os.environ["DATABRICKS_TOKEN"] = databricks_token

    # Create WorkspaceClient
    client = WorkspaceClient()
    
    # Optional check
    try:
        user_info = client.current_user.me()
        print(f"Authenticated as: {user_info.user_name}")
    except Exception as e:
        raise RuntimeError(f"WorkspaceClient authentication failed: {e}")

    return client

