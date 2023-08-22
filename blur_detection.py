## blur detection
import cv2
import numpy as np
img = cv2.imread("your_image_path")

def detect_blur_fft(image: np.array, size=60):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size : cY + size, cX - size : cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    return np.mean(magnitude)

def blur_check(img: np.array, thresh=30):      ## Value of threshold is variable, user can set it according to requiement.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = detect_blur_fft(gray, size=60)
    return bool(mean <= thresh), float(mean)


if __name__ == "__main__":
    result = blur_check(img=img)
    print("Blurness Detected: {}".format(result[0]))
    print("Threshold of Bluryness Detected: {}".format(result[1]))
