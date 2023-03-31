import os
from PIL import Image


def rename(path):
    for i, filename in enumerate(os.listdir(path)):
      if i%100 == 0: print(i)
      try:
         image = resize(os.path.join(path, filename))
         if image == None:
            continue
         image.save("./Changed_2/{:07d}.png".format(i))
         #os.remove(os.path.join(path, filename))
         #os.rename(os.path.join(path, filename), os.path.join(path, "{:05d}.png".format(i)))
      except Exception as e:
         print(e)
def resize(file):
        image = Image.open(file)
        try:
            image.verify()
        except Exception as e:
            print(1, e)
            #os.remove(file)
            return None
        else:
            image = Image.open(file)
            image = image.resize((256, 256))
            return image

rename("./Images")
