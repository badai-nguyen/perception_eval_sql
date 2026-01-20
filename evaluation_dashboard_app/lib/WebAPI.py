#!/usr/bin/python3
import json
import lib.session as session


class mapAPI:
    def __init__(self, project_id="odd2_reference"):
        self.base_url = "https://map.web.auto/v1"
        self.project_id = project_id
        self.session = session.load_session()

    def get_maps(self):
        url = f"{self.base_url}/projects/{self.project_id}/area_maps"
        return session.get_request(url, self.session)


class scenarioAPI:
    def __init__(self, project_id="odd2_reference"):
        self.base_url = "https://scenario.ci.web.auto/v1"
        self.project_id = project_id
        self.session = session.load_session()

    def get_latest_scenario(self, scenario_id):
        url = f"{self.base_url}/projects/{self.project_id}/scenarios/{scenario_id}"
        return session.get_request(url, self.session)

    def update_scenario(self, scenario_id, json_data):
        url = f"{self.base_url}/projects/{self.project_id}/scenarios/{scenario_id}"
        return session.post_request(url, self.session, data=json_data)

    def add_scenario(self, json_data):
        url = f"{self.base_url}/projects/{self.project_id}/scenarios/"
        return session.post_request(url, self.session, data=json_data)

    def delete_scenario(self, scenario_id):
        url = f"{self.base_url}/projects/{self.project_id}/scenarios/{scenario_id}/deprecate"
        return session.post_request(url, self.session)

    def get_scenario(self, scenario_id):
        url = f"{self.base_url}/projects/{self.project_id}/scenarios/{scenario_id}"
        return session.get_request(url, self.session)

    def set_approved(self, scenario_id, comment=""):
        version_id = self.get_latest_scenario(scenario_id)["version_id"]
        url = f"{self.base_url}/projects/{self.project_id}/scenarios/{scenario_id}/versions/{version_id}/review"
        json_data = json.dumps({"comment": comment, "status": "approved"})
        return session.post_request(url, self.session, data=json_data)


class evaluationAPI:
    def __init__(self, project_id="odd2_reference"):
        self.base_url = "https://evaluation.ci.web.auto/v3"
        self.project_id = project_id
        self.session = session.load_session()

    def get_suites_list(self, token=""):
        url = f"{self.base_url}/projects/{self.project_id}/suites"
        params = None
        if token != "":
            params = {"next_token": f"{token}"}
        return session.get_request(url, self.session, params=params)

    def get_suites(self, suites_id):
        url = f"{self.base_url}/projects/{self.project_id}/suites/{suites_id}"
        return session.get_request(url, self.session)

    def get_reports_list(self, token=""):
        url = f"{self.base_url}/projects/{self.project_id}/jobs/reports"
        params = None
        if token != "":
            params = {"next_token": f"{token}"}
        return session.get_request(url, self.session, params=params)

    def set_suite(self, suite_id, json):
        url = f"{self.base_url}/projects/{self.project_id}/suites/{suite_id}/versions"
        return session.post_request(url, self.session, data=json)

    def get_reports_nextlist(self, token):
        headers = {"Authorization": f"{token}"}
        url = f"{self.base_url}/projects/{self.project_id}/jobs/reports"
        return session.get_request(url, self.session, headers=headers)

    def get_report(self, report_id):
        url = f"{self.base_url}/projects/{self.project_id}/jobs/reports/{report_id}"
        return session.get_request(url, self.session)

    def get_job_report(self, job_id):
        url = f"{self.base_url}/projects/{self.project_id}/jobs/{job_id}/report"
        return session.get_request(url, self.session)

    def get_job_test_reports(self, job_id):
        url = f"{self.base_url}/projects/{self.project_id}/jobs/{job_id}/test/case/reports"
        return session.get_request(url, self.session)

    def download(self, log_id, file_name) -> str:
        url = f"{self.base_url}/projects/{self.project_id}/logs/{log_id}/download"
        schedule_json = json.dumps(
            {
                "expiration_time": 3600,
                "filename": file_name,
            }
        )
        return session.post_request(url, self.session, data=schedule_json)["url"]

    def schedule_job(self, data):
        url = f"{self.base_url}/projects/{self.project_id}/jobs/schedule"
        return session.post_request(url, self.session, data=data)

    def cancel_job(self, job_id):
        url = f"{self.base_url}/projects/{self.project_id}/jobs/{job_id}/cancel"
        return session.post_request(url, self.session)


class catalogAPI:
    def __init__(self, project_id="odd2_reference"):
        self.base_url = "https://catalog.ci.web.auto/v2"
        self.project_id = project_id
        self.session = session.load_session()

    def list_catalogs(self):
        url = f"{self.base_url}/projects/{self.project_id}/catalogs"
        return self.session.get(url)

    def update_firmware_release(self, catalog_id, firmware_release_id, release_name, description):
        url = f"{self.base_url}/projects/{self.project_id}/catalogs/{catalog_id}/firmware/releases/{firmware_release_id}"
        data_json = json.dumps(
            {
                "description": description,
                "name": release_name,
                # "request_token": self.token, # 要らない
            }
        )
        return session.patch_request(url, self.session, data_json)

    def publish_firmware_release(self, catalog_id, firmware_release_id):
        url = f"{self.base_url}/projects/{self.project_id}/catalogs/{catalog_id}/firmware/releases/{firmware_release_id}/publish"
        data_json = json.dumps(
            {
                # "request_token": self.token, # 要らない
            }
        )
        return session.post_request(url, self.session, data_json)


if __name__ == "__main__":
    api = mapAPI()
    print(api.get_maps())
