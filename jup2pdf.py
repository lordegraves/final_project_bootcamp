from os import system, mkdir, getcwd, walk
from os.path import join, splitext
from shutil import move

def jup2pdf():
    print("Please input a run_no: ")
    print("Only integers allowed")
    run_no = int(input())
    run_dir_name = 'pdf' + str(run_no)

    try:
        mkdir(run_dir_name)
    except FileExistsError:
        print(f"Directory {run_dir_name} already exists, files will be added to it.")

    for dirpath, dirnames, files in walk("."):
        for f in files:
            if f.endswith(".ipynb"):
                # Generate the full path to the current file
                full_path = join(dirpath, f)
                print(f"Converting: {full_path}")
                # Convert to PDF
                system(f"jupyter nbconvert --to pdf '{full_path}'")
                # Generate PDF filename
                f_pdf = splitext(full_path)[0] + '.pdf'
                # Move the PDF to the designated directory
                move(f_pdf, run_dir_name)

    print('The following pdf files have been created in:', run_dir_name)
    for f in listdir(run_dir_name):
        if f.endswith("pdf"):
            print(f)

if __name__ == "__main__":
    # Check for the presence of .ipynb files in all subdirectories
    ipynb_files_found = any(f.endswith(".ipynb") for dirpath, dirnames, files in walk(".") for f in files)
    if ipynb_files_found:
        print('Found Jupyter notebooks in', getcwd())
        jup2pdf()
    else:
        print('Jupyter notebooks not found in ', getcwd())
