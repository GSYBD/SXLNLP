import pandas as pd
from string import Template

html_template = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>模型对比</title>
    <style>
      table,
      thead,
      tbody,
      tr,
      th,
      td {
        margin: 0;
        padding: 0;
      }

      .container {
        width: 1200px;
        margin: 90px auto;
      }

      table {
        text-align: center;
        border-collapse: collapse;
      }

      table thead {
        background-color: #e2efda;
      }

      th {
        font-size: 16px;
      }

      th,
      td {
        border: 1px solid #000;
        padding: 6px 10px;
      }

      table tr:nth-child(2n) {
        background-color: #f2f2f2;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <table>
        $thead $tbody
      </table>
    </div>
  </body>
</html>
"""
html_template = Template(html_template)
content = pd.read_excel(r"./模型对比.xls", engine="xlrd")
thead_template = Template("<thead><tr>$th</tr></thead>")
tbody_template = Template("<tbody>$trows</tbody>")
th = ""
for i in content.columns:
    th += f"<th>{i.strip()}</th>"

trows = ""
for _, row in content.iterrows():
    tr = "<tr>"
    for c in content.columns:
        tr += f"<td>{row[c]}</td>"
    tr += "</tr>"
    trows += tr
thead = thead_template.substitute(th=th)
tbody = tbody_template.substitute(trows=trows)
print(tbody)
html = html_template.substitute(tbody=tbody, thead=thead)
with open("./模型对比.html", encoding="utf-8", mode="w") as f:
    f.write(html)
