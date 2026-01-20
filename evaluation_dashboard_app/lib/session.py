from json import JSONDecodeError
from typing import Dict

from requests import Session
from webautoauth import requests
from webautoauth.token import HttpService
from webautoauth.token import TokenSource
from webautoauth.token import load_config_from_file as load_webauto_config

HEADER = {"content-type": "application/json"}


def load_session() -> Session:
    """Get session for Web.auto authentication."""
    config = load_webauto_config()
    token_source = TokenSource(HttpService(config))
    return requests.make_session(token_source)


def get_request(url: str, session: Session, params="") -> Dict:
    """Get request to the given URL. Returns json dict."""
    if params == "":
        ret = session.get(url)
    else:
        ret = session.get(url, params=params)

    ret.raise_for_status()  # Raises HTTPError, if one occurred (e.g. 404).
    try:
        return ret.json()
    except JSONDecodeError:
        return {}


def post_request(url: str, session: Session, data=None) -> Dict:
    """Post request to the given URL. Returns json dict."""
    ret = session.post(url, headers=HEADER, data=data)
    ret.raise_for_status()  # Raises HTTPError, if one occurred (e.g. 404).
    try:
        return ret.json()
    except JSONDecodeError:
        return {}


def patch_request(url: str, session: Session, data) -> Dict:
    """Patch request to the given URL. Returns json dict."""
    ret = session.patch(url, headers=HEADER, data=data)
    ret.raise_for_status()  # Raises HTTPError, if one occurred (e.g. 404).
    try:
        return ret.json()
    except JSONDecodeError:
        return {}
