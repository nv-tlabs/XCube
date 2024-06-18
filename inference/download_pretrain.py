# import gdown
import os
import tarfile
import shutil

os.makedirs("checkpoints", exist_ok=True)

# chair
cmd = "gdown 1TtNtzzvyn0geOVdOx6wiVJPT4ZuX30FS"
os.system(cmd)
## uncompress it and put it under checkpoints
tar = tarfile.open("chair.tar")
tar.extractall()
tar.close()
shutil.move("chair", "checkpoints")
## remove the tar file
os.remove("chair.tar")

# car
cmd = "gdown 11cvrLwSn4ME1qcql4uOkBA2px8EVrvCX"
os.system(cmd)
## uncompress it and put it under checkpoints
tar = tarfile.open("car.tar")
tar.extractall()
tar.close()
shutil.move("car", "checkpoints")
## remove the tar file
os.remove("car.tar")

# plane
cmd = "gdown 1d5EH5feeKr2BVIbzB269Eh2UI0gMjLfl"
os.system(cmd)
## uncompress it and put it under checkpoints
tar = tarfile.open("plane.tar")
tar.extractall()
tar.close()
shutil.move("plane", "checkpoints")
## remove the tar file
os.remove("plane.tar")

# waymo
cmd = "gdown 1_Q_8sWDv9g1QZIZQ9R_QMQhoo741wyTK"
os.system(cmd)
## uncompress it and put it under checkpoints
tar = tarfile.open("waymo.tar")
tar.extractall()
tar.close()
shutil.move("waymo", "checkpoints")
## remove the tar file
os.remove("waymo.tar")