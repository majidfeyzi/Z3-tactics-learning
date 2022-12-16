import os


class FilesHandler:
    """
    Class to deal with smt files
    """

    @staticmethod
    def get_files_by_file_size(dirname, reverse=False):
        """ Return list of file paths in directory sorted by file size """

        # Get list of files
        filepaths = []
        for basename in os.listdir(dirname):
            filename = os.path.join(dirname, basename)
            if os.path.isfile(filename):
                filepaths.append(filename)

        # Re-populate list with filename, size tuples
        for i in range(len(filepaths)):
            filepaths[i] = (filepaths[i], os.path.getsize(filepaths[i]))

        # Sort list by file size
        # If reverse=True sort from largest to smallest
        # If reverse=False sort from smallest to largest
        filepaths.sort(key=lambda filename: filename[1], reverse=reverse)

        # Re-populate list with just filenames
        for i in range(len(filepaths)):
            filepaths[i] = (filepaths[i][0], FilesHandler.__get_size(filepaths[i][0]))

        return filepaths

    @staticmethod
    def __get_size(path):
        """
        Get file size of given file path
        :param path: file path
        :return: size in string
        """
        size = os.path.getsize(path)
        if size < 1024 * 1024:
            return f"{round(size / 1024, 2)} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{round(size / (1024 * 1024), 2)} MB"
        elif size < 1024 * 1024 * 1024 * 1024:
            return f"{round(size / (1024 * 1024 * 1024), 2)} GB"