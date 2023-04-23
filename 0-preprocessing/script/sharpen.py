from PIL import Image, ImageFilter
import glob

# path = '/media/wall/4TB_HDD/0401_new_env_all/*.png'
path = '/media/wall/4TB_HDD/full_dataset/0423_dataset/test_capsule/*.png'
for filename in glob.glob(path):
    print(filename)
    name = filename.split('/')[-1]
    print(name)
    img = Image.open(filename)

    img = img.filter(ImageFilter.SHARPEN)
    distPath = '/media/wall/4TB_HDD/full_dataset/0423_dataset/test_capsule_sharpen/' + name
    img.save(distPath)