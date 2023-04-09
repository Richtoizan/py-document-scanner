import cv2
import numpy as np
import utils

# Preprocess the image
def preprocess_image(image, width, height):
    image_resized = cv2.resize(image, (width, height))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 1)
    return image_resized, image_blur

# Apply thresholding on the image
def perform_thresholding(image_blur):
    threshold = utils.get_trackbar_values()
    image_threshold = cv2.Canny(image_blur, threshold[0], threshold[1])
    kernel = np.ones((5, 5))
    image_dilated = cv2.dilate(image_threshold, kernel, iterations=2)
    image_eroded = cv2.erode(image_dilated, kernel, iterations=1)
    return image_eroded

# Draw contours on the image
def draw_contours(image, contours):
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 10)
    return image_contours

# Warp and threshold the image
def warp_and_threshold(image, biggest, width, height):
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_warped = cv2.warpPerspective(image, matrix, (width, height))

    image_gray = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY)
    image_adaptive_threshold = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    image_adaptive_threshold = cv2.bitwise_not(image_adaptive_threshold)
    image_adaptive_threshold = cv2.medianBlur(image_adaptive_threshold, 5)

    _, image_thresholded = cv2.threshold(image_adaptive_threshold, 128, 255, cv2.THRESH_BINARY)

    return image_warped, image_gray, image_thresholded

# Perform contrast stretching on the image
def contrast_stretching(image, new_min_value=0, new_max_value=255):
    min_value = np.min(image)
    max_value = np.max(image)

    if max_value == min_value or np.isnan(min_value) or np.isnan(max_value):
        return image

    image = image.astype(np.float32)
    stretched_image = (image - min_value) * (new_max_value - new_min_value) / (max_value - min_value) + new_min_value
    return stretched_image.astype(np.uint8)

# Main function
def main():
    image_path = "1.jpg"
    height = 640
    width = 480

    utils.initialize_trackbars()
    count = 0

    while True:
        # Read and preprocess the image
        image = cv2.imread(image_path)
        image_resized, image_blur = preprocess_image(image, width, height)
        image_threshold = perform_thresholding(image_blur)

        # Find and draw contours
        contours, hierarchy = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest, max_area = utils.find_biggest_contour(contours)

        # Initialize image_biggest_contour with an empty image
        image_biggest_contour = np.zeros_like(image_resized)


        if biggest.size != 0:
            biggest = utils.reorder(biggest)
            image_biggest_contour = utils.draw_rectangle(image_resized.copy(), biggest, 2)
            image_warped, image_gray, image_adaptive_threshold = warp_and_threshold(image_resized, biggest, width, height)
            image_resized_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            image_adaptive_threshold_inverse = image_adaptive_threshold.copy()
            image_adaptive_threshold_inverse = cv2.bitwise_not(image_adaptive_threshold_inverse)
            image_gray = contrast_stretching(image_gray)


        else:

            # Initialize empty images if no contour is found
            image_warped = np.zeros_like(image_resized)
            image_gray = np.zeros_like(image_resized)
            image_adaptive_threshold = np.zeros_like(image_resized)
            image_resized_gray = np.zeros_like(image_resized)
            image_adaptive_threshold_inverse = np.zeros_like(image_resized)

        # Stack images for display
        stacked_image = utils.stack_images(([image_resized, image_biggest_contour, image_resized_gray],
                                            [image_gray, image_adaptive_threshold, image_adaptive_threshold_inverse]),
                                           0.5)

        cv2.imshow("Result", stacked_image)

        # Add text to each image
        cv2.putText(image_resized, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_biggest_contour, "Contour", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_resized_gray, "Resized Gray", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_adaptive_threshold, "Adaptive Threshold", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.putText(image_warped, "Warped", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image_gray, "Warped Gray (Scan)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image_adaptive_threshold_inverse, "Adaptive Threshold Inverse", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2)

        stacked_image = utils.stack_images(([image_resized, image_biggest_contour, image_resized_gray],
                                            [image_gray, image_adaptive_threshold, image_adaptive_threshold_inverse]),
                                           0.5)

        # Display the final result

        cv2.imshow("Result", stacked_image)

        # Save the image if 's' key is pressed

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("Scan" + str(count) + ".jpg", image_gray)

            cv2.rectangle(stacked_image,
                          ((int(stacked_image.shape[1] / 2) - 600), int(stacked_image.shape[0] / 2) + 25),
                          (800, 350), (0, 255, 0), cv2.FILLED)

            cv2.putText(stacked_image, "Scan Saved!",
                        (int(stacked_image.shape[1] / 2) - 300, int(stacked_image.shape[0] / 2)),
                        cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255))

            cv2.imshow('Result', stacked_image)
            cv2.waitKey(300)
            count += 1

if __name__ == "__main__":
    main()