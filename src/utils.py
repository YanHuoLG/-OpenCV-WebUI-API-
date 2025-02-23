import cv2

def show_image(image, window_name="Image"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()