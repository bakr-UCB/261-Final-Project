from databricks.sdk.runtime import dbutils


def lst_files_r(path):
    files = []
    dirs = [path]
    
    while dirs:
        current_dir = dirs.pop()
        for item in dbutils.fs.ls(current_dir):
            if item.isDir():
                dirs.append(item.path)
            else:
                files.append(item.path)
    
    return files

if __name__ == '__main__':
    print(lst_files_r('dbfs:/student-groups/Group_4_1'))