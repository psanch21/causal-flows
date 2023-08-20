from causal_nf.job_creator.job_creator import JobCreator


class JobCreatorShell(JobCreator):
    def __init__(self, job_folder, output_folder, header_file=None):
        super().__init__(
            job_folder, output_folder, extension="sh", header_file=header_file
        )

    def write_header(self, f):
        return

    def write_job(self, f, main_str, file_id, job_id):
        arguments = f"\npython {main_str}\n"

        f.write(arguments)
