import matplotlib
from PIL import Image

matplotlib.use('agg')

if __name__ == '__main__':
    file = 'results/nlm.png'
    img = Image.open(file)
    img = img.convert('L')

    img.save(file[0:-3] + 'jpg')
