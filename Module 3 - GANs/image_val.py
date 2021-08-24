from PIL import Image
import glob


def main():
    image_list = []
    img_number = 0
    for filename in glob.glob('data/*/*.*'):
        try:
            im = Image.open(filename)
            # print(filename)
            if str(im.mode) != "RGBA":
                print("alpha " + str(im.mode))
                img_number = img_number+1
                print(str(img_number))
        except Exception as e:
            print("Error : "+filename)


if __name__ == "__main__":
    main()