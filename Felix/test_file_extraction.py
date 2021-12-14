import os
import glob
import stat
import shutil

DATA_DIR = 'data'

# extracting images from subfolders
x_test_dir = os.path.join(DATA_DIR, 'test_set_images/')
os.chmod(x_test_dir, stat.S_IWRITE)
folder = x_test_dir
subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

for sub in subfolders:
    for f in os.listdir(sub):
        if '.png' in f:
            src = os.path.join(sub, f)
            dst = os.path.join(folder, f)
            shutil.move(src, dst)
        else:
            to_delete = os.path.join(sub, f)
            os.remove(to_delete)

# remove unnecessary files
for file in os.listdir(x_test_dir):
    if '.ini' in file:
        file_to_remove_dir = os.path.join(x_test_dir, 'desktop.ini')
        os.remove(file_to_remove_dir)
print("Test files extracted from subfolders | Other unnecessary files removed")

test_files = os.listdir(x_test_dir)
m = min(20, len(test_files))  # Load maximum 20 images
print("Loading " + str(m) + " images")
test_imgs = list(glob.iglob(x_test_dir + '*.png', recursive=True))
print(test_imgs[0])


# Extract patches from input test images
patch_size = 16  # each patch is 16*16 pixels

test_img_patches = [
    img_crop(test_imgs[i], patch_size, patch_size) for i in range(m)]
test_img_patches = np.asarray([test_img_patches[i][j] for i in range(
    len(test_img_patches)) for j in range(len(test_img_patches[i]))])
