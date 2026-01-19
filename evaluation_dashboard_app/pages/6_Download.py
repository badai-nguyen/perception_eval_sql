import streamlit as st
import json
import os
import urllib.parse
import requests
import shutil
import glob
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from typing import Text

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

def download_file(url: str, output_file: str):
    """Helper function to download files"""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

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
        for i, log_info in enumerate(log_dicts):
            # Show progress in Streamlit
            with st.spinner(f"Downloading {i+1}/{len(log_dicts)}: {log_info['scenario_name']}"):
                # Add a condition to skip if necessary
                if "second" not in log_info["scenario_name"]:
                    self.download_archive_log(log_info, "archive_id", "zip")
            remain_list.append(log_info)
        
        with st.spinner("Extracting archives..."):
            self.extract_archives(phase)
        return remain_list

    def download_result_json(self):
        log_dicts = self.get_case_simlation_log_info()
        st.write(f"Found {len(log_dicts)} result JSON files")
        
        for i, log_info in enumerate(log_dicts):
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

    def download_archive_log(self, log_info, type, format):
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
            return

        # Create output directory if it doesn't exist
        os.makedirs(self.__output_path, exist_ok=True)
        
        output_file = os.path.join(self.__output_path, dl_filename)
        download_file(content["url"], output_file)
        st.success(f"Downloaded: {dl_filename}")

    def extract_archives(self, phase):
        archive_paths = glob.glob(os.path.join(self.__output_path, "*.zip"))
        st.write(f"Found {len(archive_paths)} archives to extract")
        
        progress_bar = st.progress(0)
        for i, archive_path in enumerate(archive_paths):
            progress_bar.progress(i / len(archive_paths))
            
            dir_path = archive_path.replace(".zip", "")
            shutil.unpack_archive(archive_path, dir_path)
            os.remove(archive_path)

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


st.set_page_config(
    page_title="Autoware Evaluator Downloader",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Autoware Evaluator Results Downloader")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    environment = st.selectbox(
        "Environment",
        ["default", "dev", "stg"],
        help="Select the environment"
    )
    
    project_id = st.text_input(
        "Project ID",
        value="x2_dev",
        help="Enter the project ID"
    )
    job_id = st.text_input(
        "Job ID",
        help="Enter the job ID"
    )
    
    suite_id = st.text_input(
        "Suite ID",
        help="Enter the suite ID"
    )
    
    output_path = st.text_input(
        "Output Path",
        value="./downloads",
        help="Path where files will be saved"
    )
    
    download_type = st.radio(
        "Download Type",
        ["Archives (ZIP)", "Result JSON only"]
    )
    
    if download_type == "Archives (ZIP)":
        phase = st.text_input(
            "Phase to extract",
            value="phase_name",
            help="Enter the phase name to extract from archives"
        )

# Main content area
if st.button("Download Results", type="primary"):
    if not all([project_id, job_id, suite_id]):
        st.error("Please fill in all required fields: Project ID, Job ID, and Suite ID")
        st.stop()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
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
