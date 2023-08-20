import os.path
from abc import ABC, abstractmethod

from causal_nf.utils.io import makedirs_rm_exist, load_yaml


class JobCreator(ABC):
    def __init__(self, job_folder, output_folder, extension, header_file=None):
        job_folder = f"{job_folder}_{extension}"
        output_folder = f"{output_folder}_{extension}"
        makedirs_rm_exist(job_folder)
        makedirs_rm_exist(output_folder)

        self.extension = extension

        self.job_folder = job_folder
        self.output_folder = output_folder
        if isinstance(header_file, str) and os.path.exists(header_file):
            self.header_yaml = load_yaml(header_file)
        else:
            self.cluster_yaml = None

    @abstractmethod
    def write_header(self, filename):
        pass

    @abstractmethod
    def write_job(self, filename, main_str, file_id, job_id):
        pass

    def create_file(self, filename):
        assert not os.path.exists(filename)
        with open(filename, "a") as f:
            pass

    def _add_job(self, main_str, file_id, job_id, test=False):
        if test:
            filename = os.path.join(
                self.job_folder, f"jobs_{file_id}_test.{self.extension}"
            )
        else:
            filename = os.path.join(self.job_folder, f"jobs_{file_id}.{self.extension}")

        if not os.path.exists(filename):
            print(f"condor_submit_bid 15 {filename}")
            self.create_file(filename)
            with open(filename, "a") as f:
                self.write_header(f)
        with open(filename, "a") as f:
            self.write_job(f, main_str, file_id, job_id)

    def add_job(self, main_str, file_id, job_id):
        if job_id == 0:
            self._add_job(main_str, file_id, job_id, test=True)
        self._add_job(main_str, file_id, job_id)
