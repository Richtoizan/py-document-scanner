import cv2
import numpy as np

# Function to stack and scale images
def stack_images(img_array, scale, labels=[]):
    # Function to resize and convert grayscale images to BGR (if necessary)
    def resize_image(image, scale):
        resized_image = cv2.resize(image, (0, 0), None, scale, scale)
        if len(image.shape) == 2:
            return cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        return resized_image

    # Resize images and stack them
    img_array = [[resize_image(img, scale) for img in row] for row in img_array]
    stacked_image = np.vstack([np.hstack(row) for row in img_array])

    # Add labels to the stacked images
    if labels:
        rows, cols = len(img_array), len(img_array[0])
        each_img_width = stacked_image.shape[1] // cols
        each_img_height = stacked_image.shape[0] // rows

        for row_idx, row_labels in enumerate(labels):
            for col_idx, label in enumerate(row_labels):
                x, y = col_idx * each_img_width, row_idx * each_img_height
                cv2.rectangle(stacked_image, (x, y), (x + len(label) * 13 + 27, y + 30), (255, 255, 255), cv2.FILLED)
                cv2.putText(stacked_image, label, (x + 10, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    return stacked_image

# Function to reorder points for perspective transformation
def reorder(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    diff = np.diff(points, axis=1)

    points_new[0] = points[np.argmin(add)]
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    points_new[3] = points[np.argmax(add)]

    return points_new

# Function to find the biggest contour in a list of contours
def find_biggest_contour(contours, min_area=5000):
    biggest = np.array([])
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area

# Function to draw a rectangle around the points on an image
def draw_rectangle(img, points, thickness):
    p1, p2, p3, p4 = points[:, 0, :]
    cv2.line(img, tuple(p1), tuple(p2), (0, 255, 0), thickness)
    cv2.line(img, tuple(p1), tuple(p3), (0, 255, 0), thickness)
    cv2.line(img, tuple(p4), tuple(p2), (0, 255, 0), thickness)
    cv2.line(img, tuple(p4), tuple(p3), (0, 255, 0), thickness)

    return img

# Function to initialize trackbars
def initialize_trackbars(initial_trackbar_vals=0):
    def nothing(x):
        pass

    trackbar_window_name = "Trackbars"
    trackbar_width = 360
    trackbar_height = 120

    cv2.namedWindow(trackbar_window_name)
    cv2.resizeWindow(trackbar_window_name, trackbar_width, trackbar_height)
    cv2.createTrackbar("Lower TH", trackbar_window_name, 200, 255, nothing)
    cv2.createTrackbar("Upper TH", trackbar_window_name, 200, 255, nothing)

def get_trackbar_values():
    # Get trackbar positions for Canny thresholding
    canny_lower_threshold = cv2.getTrackbarPos("Lower TH", "Trackbars")
    canny_upper_threshold = cv2.getTrackbarPos("Upper TH", "Trackbars")
    return canny_lower_threshold, canny_upper_threshold