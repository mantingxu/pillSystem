import cv2


def take_picture(pillID, path):
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 32:
            print(count)
            img_path = path + '/' + str(count) + '.png'
            '${path}/${pillID}_${count}.png'.format(path=path, pillID=pillID, count=str(count))
            print(img_path)
            cv2.imwrite(img_path, frame)
            count += 1
        if k == 27 or count == 10:
            break


path = '/home/wall/Desktop/append'
pillID = 12370
take_picture(pillID, path)
