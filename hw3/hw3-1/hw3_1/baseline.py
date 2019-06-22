import os
import cv2
import argparse
import model
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_imgs(generator,opt):
    import matplotlib.pyplot as plt
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    noise = torch.from_numpy(noise).float().to(device)
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = model.eval_g(noise, generator).detach().cpu().numpy()
    #gen_imgs = np.interp(gen_imgs, (-1, 1), (0, 255)).astype('uint8')
    gen_imgs = (gen_imgs*128+128).astype('uint8')
    #print(gen_imgs)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(opt.output)
    plt.close()

def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))

    print("Detect {} faces".format(len(faces)))
    if len(faces) >= 20:
        print("Pass !")
    else:
       print("Fail !")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imwrite("baseline_result.png", image)
    return len(faces)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline Model')
    parser.add_argument('--input', type=str, help='Path to input image')

    args = parser.parse_args()
    detect(args.input)
