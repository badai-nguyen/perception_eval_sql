"""
Core download logic (no Streamlit) for use by the worker.
API-authenticated downloads and file organization using server-side webauto auth.
"""

import glob
import json
import logging
import os
import shutil
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)

# Default environment for evaluator API
DEFAULT_ENVIRONMENT = "default"
API_BASE_URL = "https://evaluation.ci.web.auto/v3"


def _make_evaluator_session(environment: str = DEFAULT_ENVIRONMENT):
    """Build authenticated session for evaluation.ci.web.auto API (no Streamlit)."""
    os.environ["AUTH_PROFILE"] = environment
    import webautoauth.requests
    from webautoauth.token import HttpService, TokenSource, load_config

    config = load_config()
    token_source = TokenSource(HttpService(config))
    session = webautoauth.requests.make_session(token_source)
    presigned = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    presigned.mount("http://", HTTPAdapter(max_retries=retries))
    presigned.mount("https://", HTTPAdapter(max_retries=retries))
    return session, presigned


def _is_internal_url(url: str) -> bool:
    WEBAUTO_URL = "https://evaluation.ci.web.auto/v3/"
    FMS_URL = "https://fms.web.auto/api/v1/"
    return url.startswith(WEBAUTO_URL) or url.startswith(FMS_URL)


def _download_file(
    url: str,
    output_file: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    timeout: int = 10,
    skip_large_file: bool = False,
    large_file_mb: float = 50.0,
) -> int:
    """Download URL to file. Returns bytes written. Returns 0 if skipped (large file)."""
    output_file = Path(output_file)
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    total_size = int(r.headers.get("content-length", 0))
    if skip_large_file and total_size > 0 and (total_size / (1024 * 1024)) >= large_file_mb:
        logger.info("Skipping large file %s (%.1f MB)", output_file, total_size / (1024 * 1024))
        return 0
    downloaded = 0
    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
    return downloaded


def _resp_json(resp: Any) -> Dict[str, Any]:
    """Get JSON from session get/post response (may be Response or dict)."""
    if hasattr(resp, "content"):
        return json.loads(resp.content)
    return resp


def get_case_simulation_log_info(
    auth_session: Any,
    api_base_url: str,
    project_id: str,
    job_id: str,
    suite_id: str = "",
    suite_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Paginate case/reports API and return list of log info dicts (no Streamlit)."""
    url = f"{api_base_url}/projects/{project_id}/jobs/{job_id}/test/case/reports"
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    result: List[Dict[str, Any]] = []
    next_token = ""
    while True:
        params = {"next_token": next_token, "size": 100}
        if next_token:
            resp = auth_session.get(f"{url}?{urllib.parse.urlencode(params)}", headers=headers)
        else:
            resp = auth_session.get(url, headers=headers)
        if hasattr(resp, "status_code") and resp.status_code != 200:
            raise RuntimeError(f"API error {resp.status_code}: {getattr(resp, 'content', resp)}")
        data = _resp_json(resp)
        next_token = data.get("next_token", "")
        for report in data.get("reports", []):
            suite_info = report.get("suite", {})
            sid = suite_info.get("id", "")
            sname = suite_info.get("display_name", "")
            if suite_ids:
                if sid not in suite_ids:
                    continue
            elif suite_id and sid != suite_id:
                continue
            if "simulation_archive" not in report.get("logs", {}):
                continue
            result.append({
                "suite_id": sid,
                "suite_name": sname,
                "archive_id": report["logs"]["simulation_archive"]["id"],
                "result_json_id": report["logs"]["simulation_result_json"]["id"],
                "scenario_name": report["scenario"]["display_name"],
                "scenario_id": report["scenario"]["id"],
                "scenario_ver": report["scenario"]["version_id"],
            })
        if not next_token:
            break
    return result


def _safe_path_component(value: str) -> str:
    cleaned = value.replace(os.sep, "_").replace("/", "_").strip()
    return cleaned if cleaned else "unknown"


def _get_output_path_for_log(
    output_path: str,
    log_info: Dict[str, Any],
    suite_id: str,
    suite_ids: Optional[List[str]],
) -> str:
    if suite_id and not suite_ids:
        return output_path
    sid = log_info.get("suite_id", "unknown")
    sname = log_info.get("suite_name", "")
    if sname:
        suite_dir = f"{_safe_path_component(sname)}_{_safe_path_component(sid)}"
    else:
        suite_dir = _safe_path_component(sid)
    return os.path.join(output_path, suite_dir)


def download_archive_log(
    auth_session: Any,
    presigned_session: requests.Session,
    api_base_url: str,
    project_id: str,
    log_info: Dict[str, Any],
    log_type: str,
    fmt: str,
    output_path: str,
    *,
    skip_large_file: bool = False,
    large_file_mb: float = 50.0,
) -> bool:
    """Download one log (archive or result JSON) to output_path. Returns True on success."""
    url = f"{api_base_url}/projects/{project_id}/logs/{log_info[log_type]}/download"
    post_obj = {"expiration_time": 600, "filename": "suite_log.zip"}
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    try:
        resp = auth_session.post(url, headers=headers, data=json.dumps(post_obj).encode("utf-8"))
    except Exception as e:
        logger.warning("Failed to get download URL for %s: %s", log_info.get("scenario_name"), e)
        return False
    if hasattr(resp, "status_code") and resp.status_code != 200:
        logger.warning("Failed to get download URL for %s: %s", log_info.get("scenario_name"), resp.status_code)
        return False
    content = _resp_json(resp)
    download_url = content.get("url")
    if not download_url:
        return False
    os.makedirs(output_path, exist_ok=True)
    dl_filename = log_info["scenario_name"] + "." + fmt
    output_file = os.path.join(output_path, dl_filename)
    try:
        size = _download_file(
            download_url,
            output_file,
            skip_large_file=skip_large_file,
            large_file_mb=large_file_mb,
        )
        return size > 0
    except Exception as e:
        logger.warning("Download failed %s: %s", dl_filename, e)
        return False


def extract_archives(phase: str, output_path: str, keep_zip_files: bool = False) -> None:
    """Extract zip archives under output_path, keep only the given phase and scene_result.pkl."""
    archive_paths = glob.glob(os.path.join(output_path, "*.zip"))
    for archive_path in archive_paths:
        dir_path = archive_path.replace(".zip", "")
        shutil.unpack_archive(archive_path, dir_path)
        if not keep_zip_files:
            os.remove(archive_path)
        for sub_dir_path in os.listdir(dir_path):
            if Path(sub_dir_path).name == "scenario.yaml":
                continue
            full_path = os.path.join(dir_path, sub_dir_path)
            if Path(sub_dir_path).name != phase:
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
            else:
                result_file = os.path.join(full_path, "scene_result.pkl")
                if os.path.exists(result_file):
                    shutil.move(result_file, os.path.join(dir_path, "scene_result.pkl"))
                shutil.rmtree(full_path)


def organize_files_into_directories(folder_path: str) -> None:
    """Create a directory per file (name without extension) and move file into it as result.json."""
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            new_dir_name = os.path.splitext(filename)[0]
            new_dir_path = os.path.join(folder_path, new_dir_name)
            os.makedirs(new_dir_path, exist_ok=True)
            shutil.move(file_path, os.path.join(new_dir_path, "result.json"))


def run_download_results(
    project_id: str,
    job_id: str,
    suite_id: Optional[str],
    output_path: str,
    download_type: str = "archives",
    phase: str = "first",
    *,
    skip_large_file: bool = False,
    large_file_mb: float = 50.0,
    keep_zip_files: bool = False,
    suite_ids: Optional[List[str]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Download job results (archives or result JSON), extract and organize.
    Requires server-side webauto auth (AUTH_PROFILE / ~/.webauto).
    """
    env = os.environ.get("EVALUATOR_ENVIRONMENT", DEFAULT_ENVIRONMENT)
    base_url = API_BASE_URL
    if env in ("dev", "stg"):
        base_url = f"https://scenario.ci.{env}.web.auto/v3"
    auth_session, presigned = _make_evaluator_session(env)
    suite_id = suite_id or ""
    log_dicts = get_case_simulation_log_info(
        auth_session, base_url, project_id, job_id, suite_id=suite_id, suite_ids=suite_ids
    )
    if not log_dicts:
        raise RuntimeError("No case reports found for this job/suite")
    os.makedirs(output_path, exist_ok=True)
    suite_output_paths: set = set()
    for i, log_info in enumerate(log_dicts):
        if on_progress:
            on_progress(f"Downloading {i+1}/{len(log_dicts)}: {log_info['scenario_name']}")
        if "second" in log_info.get("scenario_name", ""):
            continue
        out_dir = _get_output_path_for_log(output_path, log_info, suite_id, suite_ids)
        suite_output_paths.add(out_dir)
        if download_type == "archives":
            download_archive_log(
                auth_session,
                presigned,
                base_url,
                project_id,
                log_info,
                "archive_id",
                "zip",
                out_dir,
                skip_large_file=skip_large_file,
                large_file_mb=large_file_mb,
            )
        else:
            download_archive_log(
                auth_session,
                presigned,
                base_url,
                project_id,
                log_info,
                "result_json_id",
                "json",
                out_dir,
            )
    if on_progress:
        on_progress("Extracting archives..." if download_type == "archives" else "Organizing files...")
    for out_dir in sorted(suite_output_paths):
        if download_type == "archives":
            extract_archives(phase, out_dir, keep_zip_files=keep_zip_files)
        else:
            organize_files_into_directories(out_dir)


def run_download_scenarios(
    project_id: str,
    job_id: str,
    suite_id: Optional[str],
    output_dir: str,
    overwrite: bool = False,
    *,
    scenario_name_filter: Optional[str] = None,
    selected_ids: Optional[List[str]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Download scenarios from job to output_dir (YAML per scenario).
    Requires server-side webauto auth. After scenarios, also downloads result JSON and organizes.
    """
    import yaml
    from lib.WebAPI import scenarioAPI

    env = os.environ.get("EVALUATOR_ENVIRONMENT", DEFAULT_ENVIRONMENT)
    base_url = API_BASE_URL
    if env in ("dev", "stg"):
        base_url = f"https://scenario.ci.{env}.web.auto/v3"
    os.environ["AUTH_PROFILE"] = env
    auth_session, presigned = _make_evaluator_session(env)
    suite_id = suite_id or ""
    log_dicts = get_case_simulation_log_info(
        auth_session, base_url, project_id, job_id, suite_id=suite_id, suite_ids=None
    )
    if scenario_name_filter or selected_ids:
        filtered = []
        for log_info in log_dicts:
            if scenario_name_filter and scenario_name_filter.lower() not in log_info["scenario_name"].lower():
                continue
            if selected_ids and log_info["scenario_id"] not in selected_ids:
                continue
            filtered.append(log_info)
        log_dicts = filtered
    if not log_dicts:
        raise RuntimeError("No scenarios match the criteria")
    os.makedirs(output_dir, exist_ok=True)
    scenario_api = scenarioAPI(project_id)
    for i, log_info in enumerate(log_dicts):
        scenario_name = log_info["scenario_name"]
        scenario_id = log_info["scenario_id"]
        if on_progress:
            on_progress(f"Downloading scenario {i+1}/{len(log_dicts)}: {scenario_name}")
        out_dir = _get_output_path_for_log(output_dir, log_info, suite_id, None)
        scenario_dir = os.path.join(out_dir, scenario_name)
        yaml_path = os.path.join(scenario_dir, "scenario.yaml")
        if os.path.exists(yaml_path) and not overwrite:
            continue
        os.makedirs(scenario_dir, exist_ok=True)
        try:
            scenario = scenario_api.get_latest_scenario(scenario_id)
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    json.loads(scenario["scenario_format"]),
                    stream=f,
                    allow_unicode=True,
                    sort_keys=False,
                )
        except Exception as e:
            logger.warning("Failed to download scenario %s: %s", scenario_name, e)
            raise
    # Download result JSON files and organize (same as tab1 "Result JSON only")
    suite_output_paths = {_get_output_path_for_log(output_dir, log_info, suite_id, None) for log_info in log_dicts}
    for log_info in log_dicts:
        out_dir = _get_output_path_for_log(output_dir, log_info, suite_id, None)
        download_archive_log(
            auth_session,
            presigned,
            base_url,
            project_id,
            log_info,
            "result_json_id",
            "json",
            out_dir,
        )
    for out_dir in sorted(suite_output_paths):
        organize_files_into_directories(out_dir)
