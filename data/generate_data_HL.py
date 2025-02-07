import os
import json

folder_path = "250207_bare_hand/"
imglist = sorted(os.listdir(folder_path + "images/"))
list_ = []
for imgname in imglist:
    # because of H2ONet..
    if int(imgname[-5]) == 0:
        list_ += ["data/" + folder_path + "images/" + imgname]

print("Available images #", len(imglist))
print("Total saved images #", len(list_))

with open(folder_path + "eval_HL.json", "w") as f:
    json.dump(list_, f)