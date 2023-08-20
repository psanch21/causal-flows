import os.path

from causal_nf.utils.io import makedirs
from causal_nf.job_creator.job_creator import JobCreator


class JobCreatorSub(JobCreator):
    def __init__(self, job_folder, output_folder, header_file=None):
        super().__init__(
            job_folder, output_folder, extension="sub", header_file=header_file
        )

    def _line(self, file_id, job_id, label, extension):
        folder = os.path.join(self.output_folder, str(file_id))
        filename = os.path.join(folder, f"job_{job_id}.{extension}")
        makedirs(folder, only_if_not_exists=True)
        return f"{label} = {filename}\n"

    def error_line(self, file_id, job_id):
        return self._line(file_id, job_id, label="error", extension="err")

    def output_line(self, file_id, job_id):
        return self._line(file_id, job_id, label="output", extension="out")

    def log_line(self, file_id, job_id):
        return self._line(file_id, job_id, label="log", extension="log")

    def write_header(self, f):
        if self.header_yaml is not None:
            for key, value in self.header_yaml.items():
                my_line = f"{key} = {value}\n"
                f.write(my_line)
            f.write("\n")
        else:
            raise UserWarning("You might want to specify the cluster info")

    def write_job(self, f, main_str, file_id, job_id):
        arguments = f'\narguments = "{main_str}"\n'

        f.write(arguments)
        f.write(self.error_line(file_id, job_id))
        f.write(self.output_line(file_id, job_id))
        f.write(self.log_line(file_id, job_id))
        f.write("queue\n")
