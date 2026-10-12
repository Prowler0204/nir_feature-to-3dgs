import re

with open("/home/intern/LAI/Report/report.tex", "r") as f:
    content = f.read()

# Make images smaller
content = content.replace("[width=0.7\\textwidth]", "[width=0.45\\textwidth]")

# Change strict [H] to [htbp] to allow LaTeX to float images properly and group them
content = content.replace("\\begin{figure}[H]", "\\begin{figure}[htbp]")

with open("/home/intern/LAI/Report/report.tex", "w") as f:
    f.write(content)

print("Replacement done.")
