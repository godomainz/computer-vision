from PIL import Image
import glob


def main():
    image_list = []
    for filename in glob.glob('data/*/*.*'):
        try:
            im = Image.open(filename)
        except Exception as e:
            print("Error : "+filename)


if __name__ == "__main__":
    main()