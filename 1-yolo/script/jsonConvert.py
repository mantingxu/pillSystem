import json

# JSON file
f = open('jsonFolder/pill6.json', "r")

# Reading from file
data = json.loads(f.read())
print(len(data))
print(data[0].get('annotations')[0].get('result'))
count = 0
for i in range(0, len(data)):
    filename = data[i].get('data').get('image').split('-')[-1]
    txtName = filename.replace('.png', '').replace('.jpg','') + '.txt'
    result = data[i].get('annotations')[0].get('result')
    pillNum = len(result)
    for j in range(0, pillNum):
        x = result[j].get('value').get('x')
        y = result[j].get('value').get('y')
        width = result[j].get('value').get('width')
        height = result[j].get('value').get('height')
        label = result[j].get('value').get('rectanglelabels')[0]
        if label == 'pill':
            labelNum = 0
        else:
            labelNum = 1
        # print(x, y, width, height, labelNum)
        x_min = x
        x_max = x + width
        y_min = y
        y_max = y + height
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        x_center = x / 100.0
        y_center = y / 100.0
        yolo_w = width / 100.0
        yolo_h = height / 100.0
        # print(labelNum, x_center, y_center, yolo_w, yolo_h, filename, txtName)
        absoluteTxtPath = '/home/wall/Desktop/labels/' + txtName
        f = open(absoluteTxtPath, 'a')
        f.write(
            str(labelNum) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(yolo_w) + ' ' + str(yolo_h) + '\n')
        f.close()
        count += 1
# print(count)
# Iterating through the json
# list
# for i in data[0][]:
#     print(i)

# Closing file
# f.close()
