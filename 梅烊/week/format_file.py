import json


class FormatFileToJson:
    def __init__(self, data_path, dest_path):
        self.data_path = data_path
        self.dest_path = dest_path
        self.path = data_path

    def format_json(self):
        with open(self.dest_path ,mode='w' ,encoding="utf8") as f:
            with open(self.data_path, encoding="utf8") as fs:
                content = fs.read()
                if content.startswith('\ufeff'):
                    content = content[1:]
                lines = content.split('\n')
                for line in lines:
                    if line.strip() == "":
                        break
                    line_dict ={}
                    tag = "正面" if line.strip()[0]=='1' else "负面"
                    title = line.strip()[2:]
                    line_dict["tag"] = tag
                    line_dict["title"] = title
                    json_str = json.dumps(line_dict,ensure_ascii=False)
                    f.writelines(json_str+"\n")

if __name__ == "__main__":
    ffs =FormatFileToJson("../val_ds.csv",'../val_ds.json')
    ffs.format_json()