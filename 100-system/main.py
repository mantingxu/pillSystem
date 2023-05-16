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
folder_name = '/media/wall/4TB_HDD/full_dataset/0510_pill/pill/'

# take pictures
# openWebcam.take_picture(folder_name)

# detect
# detect_dir = pillDetection.run_detection(folder_name)
# print(detect_dir)  # dir store labels and images
# runs/detect/exp3
detect_dir = 'runs/detect/exp8'
# crop
crop_dir = cropBoundingBox.crop_pill(detect_dir, folder_name)
print(crop_dir)

# same size
[resize_dir, capsule_class_txt, pill_class_txt] = resizeCropPill.resize_pill(crop_dir, detect_dir)
print(resize_dir, capsule_class_txt, pill_class_txt)

# recognition capsule
# capsule_logger_path = capsuleDenseNet.predict_capsule_id(resize_dir, capsule_class_txt)
# print(capsule_logger_path)

# recognition pill
# pill_logger_path = pillDenseNet.predict_pill_id(resize_dir, pill_class_txt)
# print(pill_logger_path)


# draw bounding box
# drawBBox.draw_pill_bbox(pillID, detect_dir, folder_name)
