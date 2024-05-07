import os
from xml.dom.minidom import parse
from typing import List
import time


def find_covered_files(coverage_file: str = "coverage.xml"):
    data = parse(coverage_file)
    classes = data.getElementsByTagName("class")
    filenames = set(["./" + c.getAttribute("filename") for c in classes])
    return filenames


def get_extension(filename: str):
    return filename.split(".")[-1]


def find_uncovered_files(
    source_dir: str = "src",
    coverage_file: str = "coverage.xml",
    target_extensions: List[str] = ["py"],
):
    targets = set(target_extensions)
    covered = find_covered_files(coverage_file)
    not_covered = []
    for path, _, files in os.walk(f"./{source_dir}"):
        if files:
            for f in files:
                ext = get_extension(f)
                if ext in targets:
                    full_path = os.path.join(path, f)
                    if full_path not in covered:
                        not_covered.append(full_path)
    return not_covered


def html_report(
    uncovered: List[str],
    output_dir: str = "uncovered",
    output_file: str = "uncovered.html",
):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), "w") as f:
        f.write("<html><head>\n")
        f.write('<link rel="stylesheet" type="text/css" href="styles.css">\n')
        f.write("</head><body>\n")
        f.write("<h1>Uncovered Files</h1>\n")
        f.write("<table>\n")
        f.write("<tr><th>File</th></tr>\n")
        if len(uncovered) > 0:
            for file in uncovered:
                f.write(f"<tr><td>{file}</td></tr>\n")
        else:
            f.write("<tr><td>No uncovered files found</td></tr>\n")
        f.write("</table>\n")
        f.write("</body></html>\n")


def css_report(output_dir: str = "uncovered", output_file: str = "styles.css"):
    os.makedirs(output_dir, exist_ok=True)
    style = """
table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
}
tr:nth-child(even) {
    background-color: #f2f2f2;
}
th {
    padding-top: 12px;
    padding-bottom: 12px;
    text-align: left;
    background-color: #4CAF50;
    color: white;
}
"""
    with open(os.path.join(output_dir, output_file), "w") as f:
        f.write(style)


def pretty_print(uncovered: List[str], latency: float):
    heading = f"== Uncovered Files ({len(uncovered)}) found in {latency:.2f} seconds =="
    print(heading)
    print("-" * len(heading))
    for file in uncovered:
        print(file)


if __name__ == "__main__":
    start = time.time()
    uncovered = find_uncovered_files()
    pretty_print(uncovered, time.time() - start)
    html_report(uncovered)
    css_report()
