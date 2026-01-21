import streamlit as st
import json
import os
import time
import urllib.parse
import requests
import shutil
import glob
import yaml
import tempfile
import traceback
import pandas as pd
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from typing import Text, Optional, List, Dict, Any
from lib.WebAPI import scenarioAPI
from lib.perception_eval_result_summarizer import run_eval_result, generate_score_json

# --- Persistent user config helpers ---
CONFIG_FILE = os.path.expanduser("./configs/autoware_evaluator_dl_config.json")
# Constants
SCENARIO_API_BASE = "https://scenario.ci.web.auto/v1"
EVALUATION_API_BASE = "https://evaluation.ci.web.auto/v3"

def load_user_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_user_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save config: {e}")

# Initialize or load user config
_user_config = load_user_config()

def get_config_value(key, default=None):
    return _user_config.get(key, default)

def set_config_value(key, value):
    if _user_config.get(key) != value:
        _user_config[key] = value
        save_user_config(_user_config)

# Try to import the authentication module, with fallback handling
# https://github.com/tier4/webauto-auth-py.git
try:
    import webautoauth.requests
    from webautoauth.token import HttpService
    from webautoauth.token import load_config
    from webautoauth.token import TokenSource
    AUTH_AVAILABLE = True
except ImportError:
    st.warning("webautoauth not available. Authentication features will be limited.")
    AUTH_AVAILABLE = False

# Rest of your classes remain mostly the same...
class AuthcliHelper:
    def __init__(self, environment: str = "default"):
        if not AUTH_AVAILABLE:
            st.error("webautoauth is required for authentication. Please install it.")
            raise ImportError("webautoauth is not available")
            
        os.environ["AUTH_PROFILE"] = environment
        self.__headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        # web.auto resource
        config = load_config()
        token_source = TokenSource(HttpService(config))
        self.__auth_session = webautoauth.requests.make_session(token_source)
        self.__next_token = ""

        # A session to retrieve a pre-signed URL
        self.__presigned_session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        self.__presigned_session.mount("http://", HTTPAdapter(max_retries=retries))
        self.__presigned_session.mount("https://", HTTPAdapter(max_retries=retries))

    def __is_internal_url(self, url: str):
        WEBAUTO_URL = "https://evaluation.ci.web.auto/v3/"
        FMS_URL = "https://fms.web.auto/api/v1/"
        return url.startswith(WEBAUTO_URL) or url.startswith(FMS_URL)

    def get(self, url: str):
        params = {
            "next_token": self.__next_token,
            "size": 100,
        }

        # Choose session
        if self.__is_internal_url(url):
            response = self.__auth_session.get(
                f"{url}?{urllib.parse.urlencode(params)}", headers=self.__headers
            )
        else:
            # Use a session to retrieve a pre-signed URL
            response = self.__presigned_session.get(url)

        if response.status_code == 200:
            data = json.loads(response.content)
            return data
        elif response.status_code == 403:
            raise Exception(f"403 Permission denied: {response.content}")
        else:
            raise Exception(f"{response.status_code} Authorization error: {response.content}")

    def post(self, url: str, data: dict = None):
        params = {
            "next_token": self.__next_token,
            "size": 100,
        }

        # Set header
        if self.__is_internal_url(url):
            response = self.__auth_session.post(
                f"{url}?{urllib.parse.urlencode(params)}",
                headers=self.__headers,
                data=json.dumps(data).encode("utf-8"),
            )
        else:
            # Use a session to retrieve a pre-signed URL
            response = self.__presigned_session.post(url)

        if response.status_code == 200:
            data = json.loads(response.content)
            return data
        elif response.status_code == 403:
            raise Exception(f"403 Permission denied: {response.content}")
        else:
            raise Exception(f"{response.status_code} Authorization error: {response.content}")

def download_file(
    url: str,
    output_file: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    timeout: int = 10,
    min_progress_mb: float = 5.0,
    skip_large_file: bool = False,
    large_file_mb: float = 50.0,
) -> int:
    """
    Download a file with optional progress display.

    Progress is shown only if the total file size is >= min_progress_mb.

    If skip_large_file is True and file size >= large_file_mb, skip and return 0.

    Returns:
        int: downloaded size in bytes
    """
    output_file = Path(output_file)

    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()

    total_size = int(r.headers.get("content-length", 0))
    show_progress = total_size >= min_progress_mb * 1024 * 1024

    # Check for large file skip
    if skip_large_file and total_size > 0 and (total_size / (1024 * 1024)) >= large_file_mb:
        print(
            f"Skipping download of {output_file}: file size {total_size/1024/1024:.1f} MB exceeds threshold ({large_file_mb} MB)"
        )
        return 0

    downloaded = 0
    start_time = time.time()

    print(f"Downloading file to {output_file} with total size {total_size/1024/1024:.1f} MB")
    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue

            f.write(chunk)
            downloaded += len(chunk)

            if not show_progress or total_size == 0:
                continue

            elapsed = time.time() - start_time
            speed = downloaded / max(elapsed, 1e-6)
            percent = downloaded / total_size * 100
            eta = (total_size - downloaded) / max(speed, 1e-6)

            print(
                f"\r{percent:6.2f}% "
                f"({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB) "
                f"ETA {eta:5.1f}s",
                end="",
                flush=True,
            )

    if show_progress:
        print("\nDownload complete")

    return downloaded

class JobResult:
    def __init__(
        self,
        environment: Text,
        project_id: Text,
        job_id: Text,
        suite_id: Text,
        output_path: Text,
    ):
        self.__environment = environment
        self.__project_id = project_id
        self.__job_id = job_id
        self.__suite_id = suite_id
        self.__output_path = output_path
        self.__api_base_url = "https://evaluation.ci.web.auto/v3"
        if self.__environment in ["dev", "stg"]:
            self.__api_base_url = "https://scenario.ci." + self.__environment + ".web.auto/v3"

        # web resource
        self.__auth_session = AuthcliHelper(self.__environment)

    def download_archive_and_unzip(self, phase):
        log_dicts = self.get_case_simlation_log_info()

        st.write(f"Found {len(log_dicts)} logs")
        remain_list = []
        download_rows = []
        for i, log_info in enumerate(log_dicts):
            if st.session_state.get("stop_downloads"):
                st.warning("Download stopped by user.")
                break
            # Show progress in Streamlit
            with st.spinner(f"Downloading {i+1}/{len(log_dicts)}: {log_info['scenario_name']}"):
                # Add a condition to skip if necessary
                if "second" in log_info["scenario_name"]:
                    status = "skipped"
                    detail = "matched skip rule"
                else:
                    try:
                        ok = self.download_archive_log(log_info, "archive_id", "zip")
                        status = "downloaded" if ok else "failed"
                        detail = "ok" if ok else "download failed"
                    except Exception as e:
                        status = "failed"
                        detail = str(e)
                download_rows.append(
                    {
                        "archive_id": log_info["archive_id"],
                        "result_json_id": log_info["result_json_id"],
                        "scenario_name": log_info["scenario_name"],
                        "scenario_id": log_info["scenario_id"],
                        "scenario_ver": log_info["scenario_ver"],
                        "status": status,
                        "detail": detail,
                    }
                )
                remain_list.append(log_info)

        if download_rows:
            try:
                import pandas as pd

                st.subheader("📋 Download Status")
                st.dataframe(pd.DataFrame(download_rows), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render download table: {e}")
        
        with st.spinner("Extracting archives..."):
            self.extract_archives(phase)
        return remain_list

    def download_result_json(self):
        log_dicts = self.get_case_simlation_log_info()
        st.write(f"Found {len(log_dicts)} result JSON files")
        
        for i, log_info in enumerate(log_dicts):
            if st.session_state.get("stop_downloads"):
                st.warning("Download stopped by user.")
                break
            with st.spinner(f"Downloading JSON {i+1}/{len(log_dicts)}: {log_info['scenario_name']}"):
                self.download_archive_log(log_info, "result_json_id", "json")
        
        with st.spinner("Organizing files..."):
            self.organize_files_into_directories(self.__output_path)
        return log_dicts

    def get_case_simlation_log_info(self) -> list:
        simlation_archive_log_info = []
        next_token = ""
        url = (
            self.__api_base_url
            + "/projects/"
            + self.__project_id
            + "/jobs/"
            + self.__job_id
            + "/test/case/reports"
        )
        
        progress_bar = st.progress(0)
        page_count = 0
        
        while True:
            page_count += 1
            st.write(f"Fetching page {page_count}...")
            
            params = {
                "next_token": next_token,
                "size": 100,
            }
            
            if next_token != "":
                data = self.__auth_session.get(f"{url}?{urllib.parse.urlencode(params)}")
            else:
                data = self.__auth_session.get(url)
            
            next_token = data.get("next_token", "")
            
            for report in data["reports"]:
                if report["suite"]["id"] == self.__suite_id:
                    if "simulation_archive" in report["logs"]:
                        simlation_archive_log_info.append(
                            {
                                "archive_id": report["logs"]["simulation_archive"]["id"],
                                "result_json_id": report["logs"]["simulation_result_json"]["id"],
                                "scenario_name": report["scenario"]["display_name"],
                                "scenario_id": report["scenario"]["id"],
                                "scenario_ver": report["scenario"]["version_id"],
                            }
                        )
            
            # Update progress
            if next_token:
                progress_bar.progress(0.5)
            else:
                progress_bar.progress(1.0)
                break
        
        progress_bar.empty()
        return simlation_archive_log_info

    def download_archive_log(self, log_info, type, format) -> bool:
        url = (
            self.__api_base_url
            + "/projects/"
            + self.__project_id
            + "/logs/"
            + log_info[type]
            + "/download"
        )
        
        dl_filename = log_info["scenario_name"] + "." + format
        post_obj = {
            "expiration_time": 600,
            "filename": "suite_log.zip",
        }

        try:
            content = self.__auth_session.post(url, data=post_obj)
        except Exception as e:
            st.error(f"Can not get log_id {log_info[type]}: {str(e)}")
            return False

        # Create output directory if it doesn't exist
        os.makedirs(self.__output_path, exist_ok=True)
        
        output_file = os.path.join(self.__output_path, dl_filename)
        try:
            download_file(content["url"], output_file, skip_large_file=True)
        except Exception as e:
            st.error(f"Failed to download {dl_filename}: {e}")
            return False
        st.success(f"Downloaded: {dl_filename}")
        return True

    def extract_archives(self, phase):
        archive_paths = glob.glob(os.path.join(self.__output_path, "*.zip"))
        st.write(f"Found {len(archive_paths)} archives to extract")
        
        progress_bar = st.progress(0)
        for i, archive_path in enumerate(archive_paths):
            progress_bar.progress(i / len(archive_paths))
            
            dir_path = archive_path.replace(".zip", "")
            shutil.unpack_archive(archive_path, dir_path)
            #os.remove(archive_path)

            for sub_dir_path in os.listdir(dir_path):
                if Path(sub_dir_path).name == "scenario.yaml":
                    continue
                full_path = dir_path + "/" + sub_dir_path
                if not Path(sub_dir_path).name == phase:
                    if os.path.isdir(full_path):
                        shutil.rmtree(full_path)
                else:
                    result_file = os.path.join(full_path, "scene_result.pkl")
                    if os.path.exists(result_file):
                        shutil.move(
                            result_file,
                            dir_path + "/scene_result.pkl",
                        )
                    shutil.rmtree(full_path)
        
        progress_bar.progress(1.0)
        st.success("Extraction complete!")

    def organize_files_into_directories(self, folder_path):
        """
        Scan all files in the given folder, create a directory for each file,
        and move the file into its corresponding directory.
        """
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        st.write(f"Organizing {len(files)} files into directories...")
        
        progress_bar = st.progress(0)
        for i, filename in enumerate(files):
            progress_bar.progress(i / len(files))
            
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Create a new directory with the same name as the file (without extension)
                new_dir_name = os.path.splitext(filename)[0]
                new_dir_path = os.path.join(folder_path, new_dir_name)
                os.makedirs(new_dir_path, exist_ok=True)

                # Move the file into the new directory
                new_file_path = os.path.join(new_dir_path, "result.json")
                shutil.move(file_path, new_file_path)
        
        progress_bar.progress(1.0)
        st.success("File organization complete!")

    def _download_scenario_file(self, url: str, output_path: str) -> bool:
        """Download a single scenario file"""
        try:
            # Handle both direct URLs and API endpoints
            if url.startswith(("http://", "https://")):
                if self.__auth_session._AuthcliHelper__is_internal_url(url):
                    # This is an API endpoint that needs authentication
                    try:
                        content = self.__auth_session.post(url, data={"expiration_time": 600})
                        if "url" in content:
                            return download_file(content["url"], output_path)
                        else:
                            st.error(f"Unexpected response format for URL: {url}")
                            return False
                    except Exception as e:
                        st.error(f"API error for {url}: {str(e)}")
                        return False
                else:
                    # Direct URL
                    return download_file(url, output_path)
            else:
                st.error(f"Invalid URL format: {url}")
                return False
        except Exception as e:
            st.error(f"Download error for {url}: {str(e)}")
            return False
            
    def _download_additional_resources(self, yaml_content: dict, scenario_dir: str):
        """Download additional resources referenced in scenario YAML"""
        try:
            # Check for Ego vehicle
            if "ego" in yaml_content and "vehicle" in yaml_content["ego"]:
                vehicle_data = yaml_content["ego"]["vehicle"]
                if "url" in vehicle_data:
                    vehicle_url = vehicle_data["url"]
                    vehicle_file_name = os.path.basename(vehicle_url)
                    vehicle_path = os.path.join(scenario_dir, vehicle_file_name)
                    
                    if self._download_scenario_file(vehicle_url, vehicle_path):
                        yaml_content["ego"]["vehicle"]["url"] = vehicle_file_name
            
            # Check for NPC vehicles
            if "npcs" in yaml_content:
                for i, npc in enumerate(yaml_content["npcs"]):
                    if "vehicle" in npc and "url" in npc["vehicle"]:
                        npc_url = npc["vehicle"]["url"]
                        npc_file_name = os.path.basename(npc_url)
                        npc_path = os.path.join(scenario_dir, npc_file_name)
                        
                        if self._download_scenario_file(npc_url, npc_path):
                            yaml_content["npcs"][i]["vehicle"]["url"] = npc_file_name
            
            # Save updated YAML
            yaml_file_path = os.path.join(scenario_dir, "scenario.yaml")
            with open(yaml_file_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
                
        except Exception as e:
            st.warning(f"Could not download additional resources: {str(e)}")

    def download_scenarios(
        self,
        output_dir: str,
        scenario_name_filter: Optional[str] = None,
        overwrite: bool = False,
        selected_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Download scenarios from the job results.
        
        Args:
            output_dir: Directory to save downloaded scenarios
            scenario_name_filter: Optional filter for scenario names (substring match)
            overwrite: Whether to overwrite existing files
            selected_ids: Optional list of specific scenario IDs to download
            
        Returns:
            List of downloaded scenario information
        """
        log_dicts = self.get_case_simlation_log_info()
        downloaded_scenarios = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        st.write(f"Found {len(log_dicts)} scenarios in job results")
        
        # Apply filters if specified
        if scenario_name_filter or selected_ids:
            filtered_logs = []
            for log_info in log_dicts:
                include = True
                
                if scenario_name_filter and scenario_name_filter.lower() not in log_info["scenario_name"].lower():
                    include = False
                
                if selected_ids and log_info["scenario_id"] not in selected_ids:
                    include = False
                
                if include:
                    filtered_logs.append(log_info)
            
            log_dicts = filtered_logs
            st.write(f"Filtered to {len(log_dicts)} scenarios")
        
        if not log_dicts:
            st.warning("No scenarios match the filter criteria")
            return []
        
        # Download each scenario
        progress_bar = st.progress(0)
        status_text = st.empty()
        scenario_api = scenarioAPI(self.__project_id)
        for i, log_info in enumerate(log_dicts):
            progress = (i + 1) / len(log_dicts)
            progress_bar.progress(progress)
            
            scenario_name = log_info["scenario_name"]
            scenario_id = log_info["scenario_id"]
            
            status_text.text(f"Downloading scenario {i+1}/{len(log_dicts)}: {scenario_name}")
            
            scenario = scenario_api.get_latest_scenario(scenario_id) # sometimes by scenario_name
            print("scenario", scenario)
            try:
                # First get scenario metadata
                
                
                # Check if already exists
                scenario_dir = os.path.join(output_dir, f"{scenario_name}_{scenario_id}")
                yaml_file_path = os.path.join(scenario_dir, "scenario.yaml")
                scenario_dir_exists = os.path.exists(scenario_dir)
                yaml_exists = os.path.exists(yaml_file_path)

                if scenario_dir_exists and yaml_exists and not overwrite:
                    st.info(f"Skipping existing scenario: {scenario_name}")
                    downloaded_scenarios.append({
                        "name": scenario_name,
                        "id": scenario_id,
                        "path": scenario_dir,
                        "status": "skipped"
                    })
                    continue
                
                # Create scenario directory
                os.makedirs(scenario_dir, exist_ok=True)
                
                # 1. Download scenario YAML
                yaml_file_path = os.path.join(scenario_dir, "scenario.yaml")
                with open(
                    file=yaml_file_path,
                    mode="w",
                    encoding="utf-8",
                ) as f:
                    yaml.dump(
                        data=json.loads(scenario["scenario_format"]),
                        stream=f,
                        allow_unicode=True,
                        sort_keys=False,
                    )

                
                # # 2. Download map if referenced in YAML
                # try:
                #     with open(yaml_file_path, 'r') as f:
                #         yaml_content = yaml.safe_load(f)
                    
                #     if "map" in yaml_content and "url" in yaml_content["map"]:
                #         map_url = yaml_content["map"]["url"]
                #         map_file_name = os.path.basename(map_url)
                #         map_file_path = os.path.join(scenario_dir, map_file_name)
                        
                #         if not self._download_scenario_file(map_url, map_file_path):
                #             st.warning(f"Failed to download map for {scenario_name}")
                #         else:
                #             # Update YAML with local map path
                #             yaml_content["map"]["url"] = map_file_name
                #             with open(yaml_file_path, 'w') as f:
                #                 yaml.dump(yaml_content, f, default_flow_style=False)
                
                # except (yaml.YAMLError, KeyError) as e:
                #     st.warning(f"Error processing YAML for {scenario_name}: {str(e)}")
                
                # # 3. Download additional resources if needed
                # self._download_additional_resources(yaml_content, scenario_dir)
                
                downloaded_scenarios.append({
                    "name": scenario_name,
                    "id": scenario_id,
                    "path": scenario_dir,
                    "status": "success"
                })
                
                st.success(f"✓ Downloaded scenario: {scenario_name}")
                
            except Exception as e:
                st.error(f"✗ Failed to download scenario {scenario_name}: {str(e)}")
                downloaded_scenarios.append({
                    "name": scenario_name,
                    "id": scenario_id,
                    "path": "",
                    "status": "failed",
                    "error": str(e)
                })
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Summary
        st.subheader("📊 Download Summary")
        success_count = sum(1 for s in downloaded_scenarios if s["status"] == "success")
        skipped_count = sum(1 for s in downloaded_scenarios if s["status"] == "skipped")
        failed_count = sum(1 for s in downloaded_scenarios if s["status"] == "failed")
        
        st.write(f"- ✅ Successfully downloaded: {success_count}")
        st.write(f"- ⏭️ Skipped (already exists): {skipped_count}")
        st.write(f"- ❌ Failed: {failed_count}")
        
        return downloaded_scenarios

st.set_page_config(
    page_title="Autoware Evaluator Downloader",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Autoware Evaluator Results Downloader")
st.markdown("---")

# Initialize session state
if 'downloaded_scenarios' not in st.session_state:
    st.session_state.downloaded_scenarios = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Download Results"
if "stop_downloads" not in st.session_state:
    st.session_state.stop_downloads = False


def find_eval_result_dirs(root_dir: str, recursive: bool = True) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    if recursive:
        walker = os.walk(root_dir)
    else:
        walker = [(root_dir, [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))], [])]
    result_dirs = []
    for current_dir, subdirs, files in walker:
        if "scenario.yaml" in files and "scene_result.pkl" in files:
            result_dirs.append(current_dir)
    return sorted(result_dirs)


def run_eval_result_for_dir(result_dir: str, overwrite: bool = False) -> Dict[str, Any]:
    result_file = os.path.join(result_dir, "result.txt")
    score_file = os.path.join(result_dir, "score.json")
    if os.path.exists(result_file) and not overwrite:
        if os.path.exists(score_file):
            return {"path": result_dir, "status": "skipped", "detail": "result.txt exists"}
        try:
            generate_score_json(result_dir)
            return {"path": result_dir, "status": "success", "detail": "score.json generated"}
        except Exception as e:
            error_output = f"Error: {e}\n{traceback.format_exc()}"
            with open(result_file, "a", encoding="utf-8") as f:
                f.write(f"\n{error_output}")
            return {"path": result_dir, "status": "failed", "detail": str(e)}

    try:
        report_text = run_eval_result(result_dir)
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        generate_score_json(result_dir)
        return {"path": result_dir, "status": "success", "detail": "completed"}
    except Exception as e:
        error_output = f"Error: {e}\n{traceback.format_exc()}"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(error_output)
        return {"path": result_dir, "status": "failed", "detail": str(e)}


# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    environment = st.selectbox(
        "Environment",
        ["default", "dev", "stg"],
        help="Select the environment",
        index=["default", "dev", "stg"].index(get_config_value("environment", "default"))
    )
    set_config_value("environment", environment)
    
    project_id = st.text_input(
        "Project ID",
        value=get_config_value("project_id", "x2_dev"),
        help="Enter the project ID"
    )
    set_config_value("project_id", project_id)

    job_id = st.text_input(
        "Job ID",
        value=get_config_value("job_id", ""),
        help="Enter the job ID"
    )
    set_config_value("job_id", job_id)
    
    suite_id = st.text_input(
        "Suite ID",
        value=get_config_value("suite_id", ""),
        help="Enter the suite ID"
    )
    set_config_value("suite_id", suite_id)
    
    output_path = st.text_input(
        "Output Path",
        value=get_config_value("output_path", "./downloads"),
        help="Path where files will be saved"
    )
    set_config_value("output_path", output_path)
    
    download_type = st.radio(
        "Download Type",
        ["Archives (ZIP)", "Result JSON only"],
        index=["Archives (ZIP)", "Result JSON only"].index(get_config_value("download_type", "Archives (ZIP)"))
    )
    set_config_value("download_type", download_type)
    
    if download_type == "Archives (ZIP)":
        phase = st.text_input(
            "Phase to extract",
            value=get_config_value("phase", "phase_name"),
            help="Enter the phase name to extract from archives"
        )
        set_config_value("phase", phase)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📥 Download Results", "🗺️ Download Scenarios", "📊 View Downloads", "🧮 Eval Results"]
)


    
with tab1:
    st.header("Download Job Results")
        

    # Main content area
    stop_col, _ = st.columns([1, 5])
    with stop_col:
        if st.button("Stop Downloads", key="stop_downloads_btn"):
            st.session_state.stop_downloads = True
            st.warning("Stop requested. Current download will finish then halt.")


    if st.button("List Available Logs Info"):
        if not all([project_id, job_id, suite_id]):
            st.error("Please fill in all required fields: Project ID, Job ID, and Suite ID")
            st.stop()
        try:
            # Initialize JobResult
            job_result = JobResult(
                environment=environment,
                project_id=project_id,
                job_id=job_id,
                suite_id=suite_id,
                output_path=output_path
            )
            log_dicts = job_result.get_case_simlation_log_info()
            st.subheader("Simulation Logs Info")
            if log_dicts:
                display_keys = [
                    "scenario_name", 
                    "archive_id", 
                    "result_json_id", 
                    "scenario_id", 
                    "scenario_ver"
                ]
                # Flatten to DataFrame for selected keys
                table_data = [
                    {k: log.get(k, "") for k in display_keys} 
                    for log in log_dicts
                ]
                df = pd.DataFrame(table_data)
                st.dataframe(df)
            else:
                st.info("No log info found for the current suite/job/project combination.")
        except Exception as e:
            st.error(f"Failed to retrieve log info: {str(e)}")

    if st.button("Download Results", type="primary"):
        st.session_state.stop_downloads = False
        if not all([project_id, job_id, suite_id]):
            st.error("Please fill in all required fields: Project ID, Job ID, and Suite ID")
            st.stop()
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save all params to config so they're always updated with last inputs
        set_config_value("environment", environment)
        set_config_value("project_id", project_id)
        set_config_value("job_id", job_id)
        set_config_value("suite_id", suite_id)
        set_config_value("output_path", output_path)
        set_config_value("download_type", download_type)
        if download_type == "Archives (ZIP)":
            set_config_value("phase", phase)
        
        try:
            # Initialize JobResult
            job_result = JobResult(
                environment=environment,
                project_id=project_id,
                job_id=job_id,
                suite_id=suite_id,
                output_path=output_path
            )
            
            if download_type == "Archives (ZIP)":
                with st.expander("Downloading Archives", expanded=True):
                    remain_list = job_result.download_archive_and_unzip(phase)
                    st.success(f"✅ Downloaded and extracted {len(remain_list)} archives")
                    
                    # Show summary
                    st.subheader("📊 Summary")
                    st.write(f"- Total scenarios processed: {len(remain_list)}")
                    st.write(f"- Output directory: `{output_path}`")
                    
            else:  # Result JSON only
                with st.expander("Downloading Result JSON", expanded=True):
                    log_dicts = job_result.download_result_json()
                    st.success(f"✅ Downloaded {len(log_dicts)} JSON files")
                    
                    # Show summary
                    st.subheader("📊 Summary")
                    st.write(f"- Total JSON files: {len(log_dicts)}")
                    st.write(f"- Output directory: `{output_path}`")
            
            # Show file tree
            with st.expander("📁 File Structure"):
                for root, dirs, files in os.walk(output_path):
                    level = root.replace(output_path, '').count(os.sep)
                    indent = ' ' * 4 * level
                    st.text(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 4 * (level + 1)
                    for file in files:
                        st.text(f"{subindent}{file}")
                        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

    # Information section
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Get your IDs:**
            - Project ID, Job ID, and Suite ID can be found in the Autoware Evaluator URL
            - Example URL: `https://evaluation.ci.web.auto/v3/projects/{project_id}/jobs/{job_id}`
        
        2. **Choose download type:**
            - **Archives (ZIP):** Downloads zip files and extracts specific phase data
            - **Result JSON only:** Downloads only the result JSON files
        
        3. **Output:**
            - Files will be saved to the specified output directory
            - Each scenario gets its own folder
        """)


with tab2:
    st.header("Download Scenarios")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scenario_filter = st.text_input(
            "Filter by scenario name (optional)",
            placeholder="e.g., intersection, highway",
            help="Only download scenarios containing this text in their name"
        )
    
    with col2:
        overwrite = st.checkbox(
            "Overwrite existing files",
            value=False,
            help="If checked, will redownload scenarios even if they already exist"
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        st.write("Specify specific scenario IDs to download (one per line):")
        scenario_ids_text = st.text_area(
            "Scenario IDs",
            placeholder="Enter one scenario ID per line",
            height=100
        )
    
    if st.button("Download Scenarios", type="primary", key="download_scenarios"):
        if not all([project_id, job_id, suite_id]):
            st.error("Please fill in all required fields: Project ID, Job ID, and Suite ID")
            st.stop()
        
        # Parse scenario IDs
        selected_ids = None
        if scenario_ids_text:
            selected_ids = [id.strip() for id in scenario_ids_text.split('\n') if id.strip()]
            st.info(f"Will download {len(selected_ids)} specific scenarios")
        
        try:
            # Initialize JobResult
            job_result = JobResult(
                environment=environment,
                project_id=project_id,
                job_id=job_id,
                suite_id=suite_id,
                output_path=output_path
            )
            
            # Create scenarios subdirectory
            scenarios_dir = os.path.join(output_path, "scenarios")
            
            with st.expander("Downloading Scenarios", expanded=True):
                downloaded = job_result.download_scenarios(
                    output_dir=scenarios_dir,
                    scenario_name_filter=scenario_filter,
                    overwrite=overwrite,
                    selected_ids=selected_ids
                )
                
                # Store in session state
                st.session_state.downloaded_scenarios = downloaded
                
                # Show detailed results
                st.subheader("📋 Detailed Results")
                
                success_scenarios = [s for s in downloaded if s["status"] == "success"]
                if success_scenarios:
                    st.write("Successfully downloaded scenarios:")
                    for scenario in success_scenarios:
                        st.write(f"• {scenario['name']} (ID: {scenario['id']})")
                
        except Exception as e:
            st.error(f"❌ Error downloading scenarios: {str(e)}")
            st.exception(e)

with tab3:
    st.header("View Downloaded Content")
    
    if not os.path.exists(output_path):
        st.info("No downloads yet. Use the other tabs to download content first.")
    else:
        # Show directory structure
        st.subheader("📁 Directory Structure")
        
        # Let user browse directories
        selected_dir = st.selectbox(
            "Browse directory",
            [output_path] + [os.path.join(output_path, d) for d in os.listdir(output_path) 
                            if os.path.isdir(os.path.join(output_path, d))]
        )
        
        if os.path.exists(selected_dir):
            # Show files in selected directory
            st.write(f"**Files in `{selected_dir}`:**")
            
            files = os.listdir(selected_dir)
            if files:
                for file in sorted(files):
                    file_path = os.path.join(selected_dir, file)
                    if os.path.isdir(file_path):
                        st.write(f"📁 **{file}/**")
                        # Show files in subdirectory
                        sub_files = os.listdir(file_path)
                        for sub_file in sorted(sub_files)[:10]:  # Limit to 10 files
                            st.write(f"    └─ {sub_file}")
                        if len(sub_files) > 10:
                            st.write(f"    └─ ... and {len(sub_files) - 10} more files")
                    else:
                        file_size = os.path.getsize(file_path)
                        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                        st.write(f"📄 {file} ({size_str})")
            else:
                st.write("No files in this directory")
        
        # Show session state downloads
        if st.session_state.downloaded_scenarios:
            st.subheader("📋 Recent Scenario Downloads")
            df_data = []
            for scenario in st.session_state.downloaded_scenarios:
                df_data.append({
                    "Scenario Name": scenario["name"],
                    "ID": scenario["id"],
                    "Status": scenario["status"],
                    "Path": scenario.get("path", "N/A")
                })
            
            import pandas as pd
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)


with tab4:
    st.header("Eval Results (per directory)")

    eval_root = st.text_input(
        "Root directory to evaluate",
        value=get_config_value("eval_root", output_path),
        help="Directory containing downloaded scenario results",
    )
    set_config_value("eval_root", eval_root)

    col1, col2 = st.columns(2)
    with col1:
        eval_recursive = st.checkbox(
            "Search subdirectories",
            value=get_config_value("eval_recursive", True),
            help="If checked, search all subdirectories for scenario results",
        )
    with col2:
        eval_overwrite = st.checkbox(
            "Overwrite existing result.txt",
            value=get_config_value("eval_overwrite", False),
            help="If unchecked, directories with result.txt will be skipped",
        )
    set_config_value("eval_recursive", eval_recursive)
    set_config_value("eval_overwrite", eval_overwrite)

    if st.button("Run eval_result for all directories", type="primary", key="run_eval_result"):
        with st.spinner("Searching for result directories..."):
            target_dirs = find_eval_result_dirs(eval_root, recursive=eval_recursive)
            print("target_dirs", target_dirs)

        if not target_dirs:
            st.warning("No directories found with scenario.yaml and scene_result.pkl")
            st.stop()

        st.write(f"Found {len(target_dirs)} directories to process")
        progress = st.progress(0)
        results = []

        for i, result_dir in enumerate(target_dirs):
            progress.progress((i + 1) / len(target_dirs))
            results.append(run_eval_result_for_dir(result_dir, overwrite=eval_overwrite))

        progress.progress(1.0)

        success_count = sum(1 for r in results if r["status"] == "success")
        skipped_count = sum(1 for r in results if r["status"] == "skipped")
        failed_count = sum(1 for r in results if r["status"] == "failed")

        st.subheader("📊 Eval Summary")
        st.write(f"- ✅ Success: {success_count}")
        st.write(f"- ⏭️ Skipped: {skipped_count}")
        st.write(f"- ❌ Failed: {failed_count}")

        if results:
            import pandas as pd

            st.subheader("📋 Details")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
