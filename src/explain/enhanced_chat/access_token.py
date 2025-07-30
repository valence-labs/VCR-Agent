import os
import time
import webbrowser

import dotenv
import httpx
from loguru import logger

dotenv.load_dotenv()

OKTA_AUTH_ENDPOINT = os.getenv("OKTA_AUTH_ENDPOINT")
OKTA_TOKEN_ENDPOINT = os.getenv("OKTA_TOKEN_ENDPOINT")
CLIENT_ID = os.getenv("CLIENT_ID")


def get_access_token(auto_open=False):
    """
    Get access token using OAuth device code flow.

    Args:
        auto_open (bool): If True, attempt to automatically open browser.
                         If False (default), only print the URL. Safer for remote servers.

    Returns:
        str: Access token if successful, None if failed
    """
    logger.info("Starting OAuth device code flow")

    device_code_r = httpx.post(
        OKTA_AUTH_ENDPOINT,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "client_id": CLIENT_ID,
            "scope": "openid email profile groups offline_access",
        },
    )
    device_code_response = device_code_r.json()
    device_code = device_code_response["device_code"]

    # Always print the authentication URL for manual access
    logger.debug("=" * 60)
    logger.debug("AUTHENTICATION REQUIRED")
    logger.debug("=" * 60)
    logger.debug("Please open the following URL in your browser:")
    logger.debug(f"\n{device_code_response['verification_uri_complete']}\n")
    logger.debug(f"Or go to: {device_code_response['verification_uri']}")
    logger.debug(f"And enter code: {device_code_response['user_code']}")
    logger.debug("=" * 60)

    # Optionally try to open browser automatically
    if auto_open:
        try:
            logger.info("Attempting to open browser automatically")
            webbrowser.open(device_code_response["verification_uri_complete"])
            logger.success("Browser opened successfully")
        except Exception as e:
            logger.warning(f"Failed to open browser automatically: {e}")
            logger.info("Please open the URL manually")
    else:
        logger.info("Auto-open disabled - please open the URL manually")

    logger.info("Waiting for authentication...")
    access_token = None
    poll_count = 0

    # Poll until the browser authentication session has completed
    while access_token is None:
        poll_count += 1
        logger.debug(f"Polling attempt {poll_count}")

        token_r = httpx.post(
            OKTA_TOKEN_ENDPOINT,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "client_id": CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        token_r_response = token_r.json()

        if token_r_response.get("error") == "authorization_pending":
            print(".", end="", flush=True)  # Show progress dots
            time.sleep(5)
            continue
        elif token_r_response.get("access_token"):
            access_token = token_r_response.get("access_token")
            print()  # New line after dots
            logger.success("Authentication successful!")
            break
        elif token_r_response.get("error"):
            print()  # New line after dots
            error = token_r_response.get("error")
            error_desc = token_r_response.get("error_description", "Unknown error")
            logger.error(f"Authentication failed: {error}")
            logger.error(f"Description: {error_desc}")
            break
        else:
            print()  # New line after dots
            logger.error(f"Unexpected response: {token_r_response}")
            break

    return access_token
