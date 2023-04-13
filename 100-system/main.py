import create_folder as createFolder
# import webcam as openWebcam
import pill_detection as pillDetection
import crop as cropBoundingBox
import resize as resizeCropPill
import model_capsule_test as capsuleDenseNet
import model_pill_test as pillDenseNet


# import load_densenet121_all as recognitionPill
# import draw_bbox as drawBBox

# create folder
# folder_name = createFolder.folder_name()
# print(folder_name)
folder_name = './1'

# take pictures
# openWebcam.take_picture(folder_name)

# detect
detect_dir = pillDetection.run_detection(folder_name)
print(detect_dir)  # dir store labels and images
# runs/detect/exp3

# crop
crop_dir = cropBoundingBox.crop_pill(detect_dir, folder_name)
print(crop_dir)

# same size
[resize_dir, capsule_class_txt, pill_class_txt] = resizeCropPill.resize_pill(crop_dir, detect_dir)
print(resize_dir, capsule_class_txt, pill_class_txt)

# recognition capsule
capsuleDenseNet.predict_capsule_id(resize_dir, capsule_class_txt)

# recognition pill
pillDenseNet.predict_pill_id(resize_dir, pill_class_txt)



# draw bounding box
# drawBBox.draw_pill_bbox(pillID, detect_dir, folder_name)
